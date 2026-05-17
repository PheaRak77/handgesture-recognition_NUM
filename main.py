import cv2
import os
import threading
import argparse
import signal
import sys

from dotenv import load_dotenv

from src.gesture_engine import GestureEngine, GESTURE_KHMER
from src.khmer_text import put_khmer_text
from src.db import GestureDatabase

load_dotenv()

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
def open_camera(idx: int):
    # Prefer OS-native backends to avoid noisy fallbacks (e.g. FFMPEG index errors).
    backends = []
    if sys.platform.startswith("linux"):
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    elif sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        try:
            cap = cv2.VideoCapture(idx, backend)
        except Exception:
            cap = cv2.VideoCapture(idx)

        if cap is None or not cap.isOpened():
            continue

        ret, frame = cap.read()
        if ret and frame is not None:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f" Camera {idx} opened", flush=True)
            return cap

        cap.release()

    return None


def detect_all_cameras(indices=None):
    found = []
    for idx in indices or DEFAULT_CAMERA_INDICES:
        cap = open_camera(idx)
        if cap is not None:
            found.append((idx, cap))
    return found


def draw_gesture_overlay(frame, gestures, draw_khmer=True):
    h, w = frame.shape[:2]
    y = 50

    for gesture in gestures:
        khmer_word, english_desc = GESTURE_KHMER.get(
            gesture, ("មិនស្គាល់", "Unknown")
        )


        bar_h = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, y - 5), (w - 5, y + bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

   
        cv2.putText(frame, f"Gesture : {gesture}",
                    (12, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)

        cv2.putText(frame, english_desc,
                    (12, y + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if draw_khmer:
            frame = put_khmer_text(
                frame,
                f"ខ្មែរ : {khmer_word}",
                (12, y + 50),
                FONT_PATH,
                font_size=22,
                color=(0, 230, 255),
            )

        y += 95
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


def camera_worker(idx, cap, latest_frames, lock, stop_event, db=None, session_id=None):
    engine = GestureEngine()
    print(f"  📷 Camera {idx}: model ready.", flush=True)
    last_logged = None

    try:
        while not stop_event.is_set() and not global_stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f" Camera {idx}: read failed — stopping.", flush=True)
                break

            frame, gestures = engine.process_frame(frame)

            active = [g for g in gestures if g != "No Hand"]
            last_logged = _log_gesture_change(db, session_id, idx, active, last_logged)

            frame = draw_gesture_overlay(frame, active, draw_khmer=True)

            # Camera label top-right
            label = f"Cam {idx}"
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(frame, label,
                        (frame.shape[1] - tw - 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

            with lock:
                latest_frames[idx] = frame
                
    except Exception as e:
        print(f"  ❌ Camera {idx} error: {e}", flush=True)
    finally:
        print(f"  🔄 Releasing camera {idx}...", flush=True)
        cap.release()
        engine.release()
        with lock:
            latest_frames.pop(idx, None)
        print(f"  ✅ Camera {idx} released.", flush=True)


# ─────────────────────────────────────────────
# Demo mode with video file
# ─────────────────────────────────────────────
def run_demo(video_path, db=None, session_id=None):
    engine = GestureEngine()
    cap = cv2.VideoCapture(video_path)
    last_logged = None
    
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
                
            frame, gestures = engine.process_frame(frame)
            
            # Filter out "No Hand" so the overlay stays clean when no hand present
            active = [g for g in gestures if g != "No Hand"]
            last_logged = _log_gesture_change(db, session_id, 0, active, last_logged)
            frame = draw_gesture_overlay(frame, active)

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
            run_demo(args.demo, db=db, session_id=session_id)
            return

        indices = SCAN_CAMERA_INDICES if args.scan_cameras else [args.camera]
        _run_cameras(db, session_id, camera_indices=indices)
    finally:
        if db:
            db.end_session()
            db.close()
            print("  [db] Session closed.", flush=True)


def _run_cameras(db, session_id, camera_indices=None):

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

            # Check for 'q' key press (wait 1ms)
            key = cv2.waitKey(1) & 0xFF
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