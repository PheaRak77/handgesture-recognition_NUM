"""Load a trained landmark classifier (Random Forest) for CSL phrase gestures."""

from __future__ import annotations

import os

import numpy as np

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

from src.landmark_features import hands_to_feature_vector

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "gesture_classifier.joblib")


class GestureClassifier:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self._model = None
        self._labels: list[str] = []
        self._load()

    def _load(self) -> None:
        if joblib is None or not os.path.isfile(self.model_path):
            return
        payload = joblib.load(self.model_path)
        self._model = payload["model"]
        self._labels = list(payload["labels"])

    @property
    def enabled(self) -> bool:
        return self._model is not None

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def predict(self, hands_data: list[dict], *, mirror: bool = False) -> tuple[str | None, float]:
        if not self.enabled:
            return None, 0.0

        vector = hands_to_feature_vector(hands_data, mirror=mirror)
        if vector is None:
            return None, 0.0

        proba = self._model.predict_proba(np.array([vector]))[0]
        idx = int(np.argmax(proba))
        return self._labels[idx], float(proba[idx])
