# pyrefly: ignore [missing-import]
import cv2
import os
import threading
import argparse
import signal
import sys
import time
# pyrefly: ignore [missing-import]
import numpy as np

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

from src.gesture_engine import GestureEngine, GESTURE_KHMER
from src.khmer_text import put_khmer_text
from src.db import GestureDatabase

load_dotenv()

# Suppress libjpeg "Corrupt JPEG data" warnings from MJPG webcam streams.
# These are harmless (frames still decode correctly). All useful output
# goes through print() to stdout (fd 1), so redirecting stderr is safe.
if sys.platform.startswith("linux"):
    try:
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull_fd, 2)
        os.close(_devnull_fd)
    except OSError:
        pass

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
FONT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets", "fonts", "Hanuman.ttf",
)
WINDOW_TITLE_DEMO = "Khmer Gesture - Demo"
WINDOW_TITLE_CAM = "Khmer Gesture - Camera {idx}"
# Probing many indices on Windows (DSHOW) can hang several seconds per index.
DEFAULT_CAMERA_INDICES = [0]
SCAN_CAMERA_INDICES = list(range(4))

global_stop_event = threading.Event()

def signal_handler(signum, frame):
    """Handle Ctrl+C and other signals for clean shutdown."""
    print(" Received signal, shutting down gracefully...", flush=True)
    global_stop_event.set()

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ─────────────────────────────────────────────
# Open a single camera by index
# ─────────────────────────────────────────────
def _try_init_camera(idx: int, backend, force_mjpg: bool):
    try:
        cap = cv2.VideoCapture(idx, backend)
    except Exception:
        try:
            cap = cv2.VideoCapture(idx)
        except Exception:
            return None

    if cap is None or not cap.isOpened():
        return None

    # Set properties before reading any frames to ensure proper initialization
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if force_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # Disable auto-exposure priority on Linux to prevent camera dropping FPS in low light
    if sys.platform.startswith("linux"):
        import subprocess
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", f"/dev/video{idx}", "-c", "exposure_auto_priority=0", "-c", "exposure_dynamic_framerate=0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            pass

    # Read 3 frames to verify the stream is active and not failing
    success = True
    for _ in range(3):
        ret, frame = cap.read()
        if not ret or frame is None:
            success = False
            break
        time.sleep(0.01)

    if success:
        print(f" Camera {idx} opened successfully (MJPG={force_mjpg})", flush=True)
        return cap

    cap.release()
    return None


def is_video_capture_device(idx: int) -> bool:
    if not sys.platform.startswith("linux"):
        return True
    import subprocess
    try:
        # Check device capabilities using v4l2-ctl
        res = subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{idx}", "--info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1.0
        )
        if res.returncode != 0:
            return False

        # Look for Device Caps section and verify it contains "Video Capture"
        lines = res.stdout.splitlines()
        device_caps_active = False
        for line in lines:
            if "Device Caps" in line:
                device_caps_active = True
                continue
            if device_caps_active:
                if not line.strip() or ":" in line:
                    break
                if "Video Capture" in line:
                    return True
        return False
    except Exception:
        # Fallback to True if v4l2-ctl is not installed or error occurs
        return True


def open_camera(idx: int):
    # Skip Linux V4L2 metadata/control channels (e.g. /dev/video1 or /dev/video3)
    if not is_video_capture_device(idx):
        print(f" ⚠️ Skipping Camera {idx} (Metadata / Control device channel)", flush=True)
        return None

    # Prefer OS-native backends to avoid noisy fallbacks
    backends = []
    if sys.platform.startswith("linux"):
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    elif sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        # Try 1: with MJPG format (high speed for external webcams)
        cap = _try_init_camera(idx, backend, force_mjpg=True)
        if cap is not None:
            return cap

        # Try 2: default format (fallback for laptop built-in/raw webcams)
        cap = _try_init_camera(idx, backend, force_mjpg=False)
        if cap is not None:
            return cap

    return None


def detect_all_cameras(indices=None):
    found = []
    for idx in indices or DEFAULT_CAMERA_INDICES:
        cap = open_camera(idx)
        if cap is not None:
            found.append((idx, cap))
    return found


# Cache pre-rendered gesture label bars to avoid redundant blending every frame.
# key: (gesture_name, frame_width) -> annotated bar numpy array
_overlay_cache: dict[tuple, object] = {}


def draw_gesture_overlay(frame, gestures, draw_khmer=True):
    h, w = frame.shape[:2]
    y = 0

    for gesture in gestures:
        khmer_word, english_desc = GESTURE_KHMER.get(
            gesture, ("មិនស្គាល់", "Unknown")
        )

        # ── Bar geometry ────────────────────────────────────────────────────
        bar_h  = 90
        pad    = 10
        x1, y1 = pad, y - 8
        x2, y2 = w - pad, y + bar_h
        y1_c, y2_c = max(0, y1), min(h, y2)
        x1_c, x2_c = max(0, x1), min(w, x2)
        bw, bh = x2_c - x1_c, y2_c - y1_c

        cache_key = (gesture, w)
        cached_bar = _overlay_cache.get(cache_key)
        if cached_bar is None:
            bar = np.zeros((bh, bw, 3), dtype=np.uint8)

            # Accent left border (cyan-green)
            cv2.rectangle(bar, (0, 0), (5, bh), (0, 220, 160), -1)

            # Gesture name — large, bold, bright white
            cv2.putText(bar, gesture,
                        (18, 34),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

            # English description — purple
            cv2.putText(bar, english_desc,
                        (18, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 0, 200), 1, cv2.LINE_AA)

            # Khmer text — green
            if draw_khmer:
                bar = put_khmer_text(
                    bar,
                    khmer_word,
                    (bw - 200, 10),
                    FONT_PATH,
                    font_size=18,
                    color=(0, 220, 80),
                )

            _overlay_cache[cache_key] = bar
            cached_bar = bar

        # ── Blend bar onto live frame ────────────────────────────────────────
        sub = frame[y1_c:y2_c, x1_c:x2_c]
        # Darken background for contrast
        cv2.addWeighted(sub, 0.35, sub, 0.0, 0, sub)
        # Paste cached text pixels
        mask = cached_bar.any(axis=2)
        sub[mask] = cached_bar[mask]

        # Thin bottom separator line
        cv2.line(frame, (x1_c, y2_c - 1), (x2_c, y2_c - 1), (0, 120, 90), 1)

        y += bar_h + 16
    return frame


def _log_gesture_change(db, session_id, camera_id, active, last_logged):
    """Log to PostgreSQL when the primary gesture changes."""
    if db is None or not session_id:
        return last_logged

    primary = active[0] if active else None
    if primary is None:
        return None
    if primary == last_logged:
        return last_logged

    if db.log_gesture(primary, camera_id, session_id=session_id):
        print(f"  [db] Camera {camera_id}: logged '{primary}'", flush=True)
    else:
        print(
            f"  [db] Camera {camera_id}: '{primary}' not in gestures table — run seed SQL",
            flush=True,
        )
    return primary


def _init_database(use_db: bool) -> GestureDatabase | None:
    if not use_db:
        return None

    db = GestureDatabase()
    if not db.enabled:
        print("  [db] Not configured. Copy env.example to .env and set DB_NAME.", flush=True)
        return None

    try:
        info = db.test_connection()
        print(
            f"  [db] Connected to '{info['db']}' "
            f"({info['gesture_count']} gestures in catalog).",
            flush=True,
        )
        if info["gesture_count"] == 0:
            print(
                "  [db] gestures table is empty — run database/02_seed_gestures.sql in pgAdmin.",
                flush=True,
            )
        return db
    except Exception as exc:
        print(f"  [db] Connection failed: {exc}", flush=True)
        print("  [db] Continuing without database.", flush=True)
        return None


def camera_worker(
    idx,
    cap,
    latest_frames,
    lock,
    stop_event,
    db=None,
    session_id=None,
    *,
    metrics: bool = False,
):
    """
    3-thread pipeline for maximum FPS:
      Thread A (capture)   – reads frames as fast as the camera outputs them,
                             always overwrites a single-slot buffer with the newest frame.
      Thread B (inference) – this thread; pulls the latest frame, runs MediaPipe
                             + gesture detection, pushes the annotated frame to display.
    Decoupling capture from inference means cap.read() never blocks TFLite
    and TFLite never starves the camera buffer.
    """
    import queue as _queue

    # Single-slot buffer: maxsize=1 means if inference is busy the old frame
    # is discarded and only the newest frame is kept.
    raw_slot: _queue.Queue = _queue.Queue(maxsize=1)

    # ── Thread A: capture ────────────────────────────────────────────────────
    def _capture_loop():
        while not stop_event.is_set() and not global_stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f" Camera {idx}: read failed — stopping.", flush=True)
                stop_event.set()
                break
            # Drop stale frame so inference always gets the newest one.
            try:
                raw_slot.get_nowait()
            except _queue.Empty:
                pass
            try:
                raw_slot.put_nowait(frame)
            except _queue.Full:
                pass  # inference just grabbed it; next loop will refill

    capture_thread = threading.Thread(target=_capture_loop,
                                      name=f"Cap-{idx}", daemon=True)

    # ── Thread B: inference ──────────────────────────────────────────────────
    engine = GestureEngine(camera_id=idx)
    print(f"  📷 Camera {idx}: model ready.", flush=True)

    last_logged = None
    fps_frames = 0
    fps_t0 = time.perf_counter()
    fps_value = 0.0
    lat_ms_avg = 0.0

    capture_thread.start()

    try:
        while not stop_event.is_set() and not global_stop_event.is_set():
            try:
                frame = raw_slot.get(timeout=0.05)
            except _queue.Empty:
                continue

            t_start = time.perf_counter() if metrics else 0.0
            frame, gestures = engine.process_frame(frame)

            active = [g for g in gestures if g != "No Hand"]
            last_logged = _log_gesture_change(db, session_id, idx, active, last_logged)

            frame = draw_gesture_overlay(frame, active, draw_khmer=True)

            with lock:
                latest_frames[idx] = frame

    except Exception as e:
        print(f"  ❌ Camera {idx} error: {e}", flush=True)
    finally:
        print(f"  🔄 Releasing camera {idx}...", flush=True)
        capture_thread.join(timeout=2)
        cap.release()
        engine.release()
        with lock:
            latest_frames.pop(idx, None)
        print(f"  ✅ Camera {idx} released.", flush=True)



# ─────────────────────────────────────────────
# Demo mode with video file
# ─────────────────────────────────────────────
def run_demo(video_path, db=None, session_id=None, *, metrics: bool = False):
    engine = GestureEngine(camera_id=0)
    cap = cv2.VideoCapture(video_path)
    last_logged = None
    fps_frames = 0
    fps_t0 = time.perf_counter()
    fps_value = 0.0
    lat_ms_avg = 0.0
    
    if not cap.isOpened():
        print(f"❌ Could not open video file: {video_path}")
        return
    
    cv2.namedWindow(WINDOW_TITLE_DEMO, cv2.WINDOW_NORMAL)
    print(" Supported gestures:", ", ".join(GESTURE_KHMER.keys()))
    print(" Controls:")
    print("   • Press 'q' in the demo window to quit")
    print("   • Or press Ctrl+C in terminal to quit")
    print("   • Click on the window first, then press 'q'\n", flush=True)
    
    try:
        while not global_stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("🎬 Video ended. Restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            t_start = time.perf_counter() if metrics else 0.0
            frame, gestures = engine.process_frame(frame)
            
            # Filter out "No Hand" so the overlay stays clean when no hand present
            active = [g for g in gestures if g != "No Hand"]
            last_logged = _log_gesture_change(db, session_id, 0, active, last_logged)
            frame = draw_gesture_overlay(frame, active)

            if metrics:
                t_end = time.perf_counter()
                lat_ms = (t_end - t_start) * 1000.0
                lat_ms_avg = (0.9 * lat_ms_avg + 0.1 * lat_ms) if lat_ms_avg else lat_ms

                fps_frames += 1
                now = t_end
                dt = now - fps_t0
                if dt >= 1.0:
                    fps_value = fps_frames / dt
                    fps_frames = 0
                    fps_t0 = now

                cv2.putText(
                    frame,
                    f"FPS: {fps_value:.1f}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Latency: {lat_ms_avg:.1f} ms",
                    (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )

            # Demo label
            cv2.putText(frame, "Demo Mode",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow(WINDOW_TITLE_DEMO, frame)
            
            # Check for 'q' key press (wait 1ms)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("👋 'q' pressed, shutting down...", flush=True)
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt, shutting down...", flush=True)
    except Exception as e:
        print(f"\n❌ Error in demo loop: {e}", flush=True)
    
    cap.release()
    engine.release()
    cv2.destroyAllWindows()
    print("👋 Demo ended. Goodbye!", flush=True)


# ─────────────────────────────────────────────
# Main — GUI in main thread only
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Khmer Gesture Recognition")
    parser.add_argument("--demo", type=str, help="Run in demo mode with a video file")
    parser.add_argument(
        "--db",
        action="store_true",
        help="Log gesture detections to PostgreSQL (uses .env settings)",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_false",
        dest="metrics",
        help="Hide FPS + inference latency overlay on video",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to use (default: 0)",
    )
    parser.add_argument(
        "--scan-cameras",
        action="store_true",
        help="Try camera indices 0-3 (slower on Windows; default uses --camera only)",
    )
    args = parser.parse_args()

    use_db = args.db or (
        os.getenv("ENABLE_DB", "").lower() in ("1", "true", "yes")
    )
    db = _init_database(use_db)
    session_id = db.session_id if db else None

    try:
        if args.demo:
            print(f"🎬 Running in demo mode with video: {args.demo}")
            if not os.path.exists(args.demo):
                print(f"❌ Video file not found: {args.demo}")
                return
            if db:
                session_id = db.start_session(mode="demo", camera_count=1)
            run_demo(args.demo, db=db, session_id=session_id, metrics=args.metrics)
            return

        indices = SCAN_CAMERA_INDICES if args.scan_cameras else [args.camera]
        _run_cameras(db, session_id, camera_indices=indices, metrics=args.metrics)
    finally:
        if db:
            db.end_session()
            db.close()
            print("  [db] Session closed.", flush=True)


def _run_cameras(db, session_id, camera_indices=None, *, metrics: bool = False):

    print("🔍 Opening camera…", flush=True)
    cameras = detect_all_cameras(camera_indices)

    if not cameras:
        print("❌ No working camera found.")
        print("   Tried indices:", ", ".join(map(str, camera_indices or DEFAULT_CAMERA_INDICES)))
        print("   Please connect a camera and try again.")
        print("   Or run in demo mode with a video file: python main.py --demo video.mp4")
        return

    if db:
        session_id = db.start_session(mode="live", camera_count=len(cameras))

    print(f"\n📸 Found {len(cameras)} camera(s): "
          + ", ".join(f"index {i}" for i, _ in cameras))
    print("✋ Supported gestures:", ", ".join(GESTURE_KHMER.keys()))
    print("🎮 Controls:")
    print("   • Press 'q' in any camera window to quit")
    print("   • Or press Ctrl+C in terminal to quit")
    print("   • Click on a camera window first, then press 'q'\n", flush=True)

    for idx, _ in cameras:
        cv2.namedWindow(WINDOW_TITLE_CAM.format(idx=idx), cv2.WINDOW_NORMAL)

    latest_frames: dict = {}
    lock = threading.Lock()
    stop_event = threading.Event()

    threads = []
    for idx, cap in cameras:
        t = threading.Thread(
            target=camera_worker,
            args=(idx, cap, latest_frames, lock, stop_event, db, session_id),
            kwargs={"metrics": metrics},
            daemon=True,
            name=f"Cam-{idx}",
        )
        threads.append(t)
        t.start()

    # ── Main display loop (must run on main thread for OpenCV UI) ─────────
    try:
        while not stop_event.is_set() and not global_stop_event.is_set():
            with lock:
                snapshot = dict(latest_frames)

            if snapshot:
                for idx, frame in snapshot.items():
                    cv2.imshow(WINDOW_TITLE_CAM.format(idx=idx), frame)
            else:
                # First frames still processing (MediaPipe load) — keep UI alive
                cv2.waitKey(1)

            # Wait 2ms to pump GUI events and update window with zero lag
            key = cv2.waitKey(2) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("👋 'q' pressed, shutting down...", flush=True)
                stop_event.set()
                break

    except KeyboardInterrupt:
        print("\\ Keyboard interrupt, shutting down...", flush=True)
        stop_event.set()
    except Exception as e:
        print(f"\n Error in main loop: {e}", flush=True)
        stop_event.set()

    # Cleanup
    stop_event.set()
    print("⏳ Waiting for camera threads to finish...", flush=True)
    for t in threads:
        t.join(timeout=5)
        if t.is_alive():
            print(f"  Thread {t.name} didn't finish cleanly")

    cv2.destroyAllWindows()
    print("👋 All cameras closed. Goodbye!", flush=True)


if __name__ == "__main__":
    main()