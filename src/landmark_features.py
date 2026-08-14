"""Normalize MediaPipe hand landmarks into camera-invariant feature vectors."""

from __future__ import annotations

# pyrefly: ignore [missing-import]
import numpy as np

FEATURES_PER_HAND = 63  # 21 landmarks × (x, y, z)
MAX_HANDS = 2
FEATURE_DIM = FEATURES_PER_HAND * MAX_HANDS
WINDOW_SIZE = 10
ROLLING_FEATURE_DIM = FEATURE_DIM * WINDOW_SIZE


def normalize_hand_landmarks(landmarks) -> np.ndarray:
    """Wrist-relative coords scaled by palm size (wrist → middle MCP)."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = float(np.linalg.norm(pts[9, :2]))
    if scale < 1e-6:
        scale = 1.0
    pts[:, :2] /= scale
    return pts


def _hand_sort_key(hand: dict, mirror: bool) -> float:
    x = hand["landmarks"][0].x
    return -x if mirror else x


def hands_to_feature_vector(hands_data: list[dict], *, mirror: bool = False) -> np.ndarray | None:
    """Fixed 126-d vector: up to two hands, left-to-right (optionally mirrored)."""
    if not hands_data:
        return None

    ordered = sorted(hands_data, key=lambda h: _hand_sort_key(h, mirror))
    chunks: list[float] = []

    for hand in ordered[:MAX_HANDS]:
        pts = normalize_hand_landmarks(hand["landmarks"])
        if mirror:
            pts[:, 0] *= -1.0
        chunks.extend(pts.flatten().tolist())

    while len(chunks) < FEATURE_DIM:
        chunks.extend([0.0] * FEATURES_PER_HAND)

    return np.array(chunks[:FEATURE_DIM], dtype=np.float32)


def feature_vector_to_row(label: str, vector: np.ndarray) -> list:
    return [label, *vector.tolist()]
