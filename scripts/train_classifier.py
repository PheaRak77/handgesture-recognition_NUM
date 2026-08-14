#!/usr/bin/env python3
"""Train a Random Forest classifier on collected landmark CSV files."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.landmark_features import FEATURE_DIM

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "landmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_classifier.joblib")


def load_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files in {data_dir}. Run scripts/collect_landmarks.py first.")

    rows_x: list[list[float]] = []
    rows_y: list[str] = []

    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                label = row["label"].strip()
                features = [float(row[f"f{i}"]) for i in range(FEATURE_DIM)]
                if not features:
                    continue
                rows_x.append(features)
                rows_y.append(label)

    if not rows_x:
        raise ValueError("CSV files found but no sample rows.")

    return np.array(rows_x, dtype=np.float32), np.array(rows_y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train gesture landmark classifier")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Folder with *.csv landmark files")
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out fraction")
    args = parser.parse_args()

    x, y = load_dataset(args.data_dir)
    labels = sorted(set(y.tolist()))
    print(f"Loaded {len(x)} samples, {len(labels)} classes")
    print("  classes:", ", ".join(labels))

    if len(x) < 20:
        print("Warning: very few samples — aim for 50+ per gesture for good accuracy.")

    _, counts = np.unique(y, return_counts=True)
    stratify = y if len(counts) > 1 and counts.min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=42, stratify=stratify
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=24,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print("\nValidation report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model, "labels": labels}, MODEL_PATH)
    print(f"\nSaved model → {MODEL_PATH}")
    print("Run: python main.py --camera 0")


if __name__ == "__main__":
    main()
