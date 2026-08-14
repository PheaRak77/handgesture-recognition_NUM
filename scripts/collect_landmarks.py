#!/usr/bin/env python3
"""Record normalized hand landmarks for training the gesture classifier."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.camera_config import load_camera_config
from src.gesture_engine import GestureEngine
from src.landmark_features import FEATURE_DIM, feature_vector_to_row

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "landmarks")
CSV_HEADER = ["label"] + [f"f{i}" for i in range(FEATURE_DIM)]

# CSL / phrase gestures first — best ROI for training
PHRASE_GESTURES = [
    "Hello", "How Are You", "Where From", "Thank You", "Please",
    "Right", "Wrong", "Understand", "Again", "Deaf", "Congratulation", "Hearing",
    "Drink", "Delicious", "Beautiful",
    "Like", "I, me", "You", "Happy", "Today", "People", "Participate", "Age", "All of you",
]
SIMPLE_GESTURES = [
    "Fist", "Open Hand", "Thumbs Up", "Thumbs Down", "One", "Two", "Three",
    "Four", "Five", "OK", "Call Me", "Point Up",
]
RECORDABLE = PHRASE_GESTURES + SIMPLE_GESTURES + ["No Hand"]


def _gesture_csv_path(label: str) -> str:
    safe = label.replace(" ", "_").lower()
    return os.path.join(DATA_DIR, f"{safe}.csv")


def _sample_vector(hands, label: str, mirror: bool):
    from src.landmark_features import hands_to_feature_vector

    if label == "No Hand":
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    if not hands:
        return None
    return hands_to_feature_vector(hands, mirror=mirror)


def _append_sample(label: str, vector) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _gesture_csv_path(label)
    write_header = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(CSV_HEADER)
        writer.writerow(feature_vector_to_row(label, vector))


def _draw_ui(frame, label: str, samples: int, cfg: dict, recording: bool) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 72), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Label: {label}  |  samples: {samples}  |  mirror={cfg['mirror']}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 120),
        2,
    )
    status = "RECORDING" if recording else "Hold pose — press SPACE"
    cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    cv2.putText(
        frame,
        "[ / ] prev/next  SPACE=30 frames  n=no-hand  q=quit",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect hand landmark training data")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--batch", type=int, default=30, help="Frames per SPACE press")
    args = parser.parse_args()

    cfg = load_camera_config(args.camera)
    engine = GestureEngine(camera_id=args.camera)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        sys.exit(1)

    idx = 0
    label = RECORDABLE[idx]
    total_samples = 0
    recording_until = 0.0

    print("Collect landmarks for ML training")
    print(f"  camera : {args.camera}")
    print(f"  output : {DATA_DIR}")
    print(f"  gestures: {len(RECORDABLE)} labels")
    print("  Controls: [ ] switch label | SPACE record | n = No Hand | q quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, _ = engine.process_frame(frame)
            hands = engine.last_hands_data
            now = time.time()
            recording = now < recording_until

            if recording:
                vector = _sample_vector(hands, label, cfg["mirror"])
                if vector is not None:
                    _append_sample(label, vector)
                    total_samples += 1

            _draw_ui(frame, label, total_samples, cfg, recording)
            cv2.imshow("Collect Landmarks", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("["):
                idx = (idx - 1) % len(RECORDABLE)
                label = RECORDABLE[idx]
            elif key == ord("]"):
                idx = (idx + 1) % len(RECORDABLE)
                label = RECORDABLE[idx]
            elif key == ord(" "):
                recording_until = now + 0.5
                print(f"  Recording '{label}' (~{args.batch} frames)...", flush=True)
                for _ in range(args.batch):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    engine.process_frame(frame)
                    hands = engine.last_hands_data
                    vector = _sample_vector(hands, label, cfg["mirror"])
                    if vector is not None:
                        _append_sample(label, vector)
                        total_samples += 1
                    cv2.waitKey(1)
                print(f"  Done. Total samples: {total_samples}", flush=True)
            elif key == ord("n"):
                label = "No Hand"
                idx = -1
    finally:
        cap.release()
        engine.release()
        cv2.destroyAllWindows()
        print(f"\nSaved {total_samples} samples under {DATA_DIR}")
        print("Next: python scripts/train_classifier.py")


if __name__ == "__main__":
    main()
