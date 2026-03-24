import cv2
import os
import threading
import argparse
import signal
import sys
from src.gesture_engine import GestureEngine, GESTURE_KHMER
from src.khmer_text import put_khmer_text

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
FONT_PATH = os.path.join("assets", "fonts", "Hanuman.ttf")
PROBE_INDICES = list(range(8))

# Global stop event for clean shutdown
global_stop_event = threading.Event()

def signal_handler(signum, frame):
    """Handle Ctrl+C and other signals for clean shutdown."""
    print("\n🛑 Received signal, shutting down gracefully...", flush=True)
    global_stop_event.set()

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ─────────────────────────────────────────────
# Open a single camera by index
# ─────────────────────────────────────────────
def open_camera(idx: int):
    for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
        try:
            cap = cv2.VideoCapture(idx, backend)
        except Exception:
            cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and float(frame.mean()) > 5.0:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"  ✅ Camera {idx} opened", flush=True)
                return cap
            cap.release()
    return None


def detect_all_cameras():
    found = []
    for idx in PROBE_INDICES:
        cap = open_camera(idx)
        if cap is not None:
            found.append((idx, cap))
    return found


# ─────────────────────────────────────────────
# Overlay helper — draws the gesture info box
# ─────────────────────────────────────────────
def draw_gesture_overlay(frame, gestures):
    h, w = frame.shape[:2]
    y = 50

    for gesture in gestures:
        khmer_word, english_desc = GESTURE_KHMER.get(
            gesture, ("មិនស្គាល់", "Unknown")
        )

        # Semi-transparent background bar
        bar_h = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, y - 5), (w - 5, y + bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # English gesture name (green)
        cv2.putText(frame, f"Gesture : {gesture}",
                    (12, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)

        # English description (white)
        cv2.putText(frame, english_desc,
                    (12, y + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Khmer text (yellow) — uses PIL for proper Khmer rendering
        frame = put_khmer_text(
            frame,
            f"ខ្មែរ : {khmer_word}",
            (12, y + 50),
            FONT_PATH,
            font_size=22,
            color=(0, 230, 255),
        )

        y += 95  # move down for next hand

    return frame


# ─────────────────────────────────────────────
# Per-camera background worker
# ─────────────────────────────────────────────
def camera_worker(idx, cap, latest_frames, lock, stop_event):
    engine = GestureEngine()
    print(f"  📷 Camera {idx}: model ready.", flush=True)

    try:
        while not stop_event.is_set() and not global_stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"  ⚠️  Camera {idx}: read failed — stopping.", flush=True)
                break

            frame, gestures = engine.process_frame(frame)

            # Filter out "No Hand" so the overlay stays clean when no hand present
            active = [g for g in gestures if g != "No Hand"]
            frame = draw_gesture_overlay(frame, active)

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
def run_demo(video_path):
    engine = GestureEngine()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open video file: {video_path}")
        return
    
    cv2.namedWindow("Khmer Gesture — Demo", cv2.WINDOW_NORMAL)
    print("✋ Supported gestures:", ", ".join(GESTURE_KHMER.keys()))
    print("🎮 Controls:")
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
            frame = draw_gesture_overlay(frame, active)
            
            # Demo label
            cv2.putText(frame, "Demo Mode",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Khmer Gesture — Demo", frame)
            
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
    args = parser.parse_args()

    if args.demo:
        print(f"🎬 Running in demo mode with video: {args.demo}")
        if not os.path.exists(args.demo):
            print(f"❌ Video file not found: {args.demo}")
            return
        run_demo(args.demo)
        return

    print("🔍 Scanning for available cameras…", flush=True)
    cameras = detect_all_cameras()

    if not cameras:
        print("❌ No working camera found.")
        print("   Tried indices:", ", ".join(map(str, PROBE_INDICES)))
        print("   Please connect a camera and try again.")
        print("   Or run in demo mode with a video file: python main.py --demo video.mp4")
        return

    print(f"\n📸 Found {len(cameras)} camera(s): "
          + ", ".join(f"index {i}" for i, _ in cameras))
    print("✋ Supported gestures:", ", ".join(GESTURE_KHMER.keys()))
    print("🎮 Controls:")
    print("   • Press 'q' in any camera window to quit")
    print("   • Or press Ctrl+C in terminal to quit")
    print("   • Click on a camera window first, then press 'q'\n", flush=True)

    for idx, _ in cameras:
        cv2.namedWindow(f"Khmer Gesture — Camera {idx}", cv2.WINDOW_NORMAL)

    latest_frames: dict = {}
    lock = threading.Lock()
    stop_event = threading.Event()

    threads = []
    for idx, cap in cameras:
        t = threading.Thread(
            target=camera_worker,
            args=(idx, cap, latest_frames, lock, stop_event),
            daemon=True,
            name=f"Cam-{idx}",
        )
        threads.append(t)
        t.start()

    # ── Main display loop ──────────────────────────────────────────────────
    try:
        while not stop_event.is_set() and not global_stop_event.is_set():
            with lock:
                snapshot = dict(latest_frames)

            for idx, frame in snapshot.items():
                cv2.imshow(f"Khmer Gesture — Camera {idx}", frame)

            # Check for 'q' key press (wait 1ms)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("👋 'q' pressed, shutting down...", flush=True)
                stop_event.set()
                break

    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt, shutting down...", flush=True)
        stop_event.set()
    except Exception as e:
        print(f"\n❌ Error in main loop: {e}", flush=True)
        stop_event.set()

    # Cleanup
    stop_event.set()
    print("⏳ Waiting for camera threads to finish...", flush=True)
    for t in threads:
        t.join(timeout=5)
        if t.is_alive():
            print(f"⚠️  Thread {t.name} didn't finish cleanly")

    cv2.destroyAllWindows()
    print("👋 All cameras closed. Goodbye!", flush=True)


if __name__ == "__main__":
    main()