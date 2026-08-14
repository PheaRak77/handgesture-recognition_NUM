import cv2
import os
import time
# Tell TFLite XNNPACK to use all available CPU cores for faster inference
os.environ.setdefault("TFLITE_NUM_THREADS", "4")
os.environ.setdefault("XNNPACK_NUM_THREADS", "4")
from collections import deque
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.camera_config import load_camera_config
from src.gesture_classifier import GestureClassifier

GESTURE_KHMER = {
    "No Hand":        ("គ្មានដៃ",        "No hand detected"),
    "Fist":           ("ក្ដាប់ដៃ",       "Fist / Stop"),
    "Open Hand":      ("ដៃបើក",         "Open hand / Hello"),
    "Thumbs Up":      ("ល្អណាស់",        "Thumbs Up / Good"),
    "Thumbs Down":    ("មិនល្អ",         "Thumbs Down / Bad"),
    "One":            ("មួយ",           "Number One / Index finger"),
    "Two":            ("ពីរ",            "Number Two"),
    "Three":          ("បី",            "Number Three"),
    "Four":           ("បួន",           "Number Four"),
    "Five":           ("ប្រាំ",          "Number Five"),
    "OK":             ("យល់ព្រម",        "OK / Agree"),
    "Call Me":        ("ហៅខ្ញុំ",        "Call me / Phone"),
    "Point Up":       ("ចង្អុលឡើង",      "Point Up / Attention"),

    # ── CSL phrases (reference charts) ─────────────────────────────────────
    "Hello":          ("សួស្តី",         "Hello / Greeting"),
    "How Are You":    ("សុខសប្បាយ",      "How are you?"),
    "Where From":     ("មកពីណា",        "Where are you from?"),
    "Thank You":      ("អរគុណ",         "Thank you"),
    "Please":         ("សូម",           "Please"),
    "Right":          ("ត្រូវ",          "Right"),
    "Wrong":          ("ខុស",            "Wrong"),
    "Understand":     ("យល់",            "Understand"),
    "Again":          ("ម្តងទៀត",        "Again"),
    "Deaf":           ("ថ្លង់",           "Deaf"),
    "Congratulation": ("អបអរសាទរ",       "Congratulation"),
    "Hearing":        ("ស្តាប់ឮ",         "Hearing"),
    "Drink":          ("ផឹកទឹក",         "Drink"),
    "Delicious":      ("រសជាតិឆ្ងាញ់",         "Delicious"),
    "Beautiful":      ("ស្អាត",          "Beautiful"),
    "Like":           ("ចូលចិត្ត",        "Like"),
    "I, me":          ("ខ្ញុំ",           "I, me"),
    "You":            ("អ្នក",           "You"),
    "Happy":          ("សប្បាយ",         "Happy"),
    "Today":          ("ថ្ងៃនេះ",         "Today"),
    "People":         ("មនុស្ស",         "People"),
    "Participate":    ("ចូលរួម",         "Participate, join"),
    "Age":            ("អាយុ",          "Age"),
    "All of you":     ("ទាំងអស់គ្នា",     "All of you"),
}

_OPEN_PALM = frozenset({"Open Hand", "Five"})


def _is_open_palm(gesture: str) -> bool:
    return gesture in _OPEN_PALM


def _wrist_distance(lm1, lm2) -> float:
    return ((lm1[0].x - lm2[0].x) ** 2 + (lm1[0].y - lm2[0].y) ** 2) ** 0.5


def _tip_distance(lm1, idx1, lm2, idx2) -> float:
    return ((lm1[idx1].x - lm2[idx2].x) ** 2 + (lm1[idx1].y - lm2[idx2].y) ** 2) ** 0.5


def _palms_facing_up(lm) -> bool:
    wrist_y = lm[0].y
    tips = (8, 12, 16, 20)
    above = sum(1 for i in tips if lm[i].y < wrist_y - 0.04)
    return above >= 3


def _finger_spread(lm) -> float:
    return ((lm[8].x - lm[20].x) ** 2 + (lm[8].y - lm[20].y) ** 2) ** 0.5


def _index_only(lm, hand_label: str) -> bool:
    thumb, index, middle, ring, pinky = _finger_states_static(lm, hand_label)
    return index and not middle and not ring and not pinky


def _index_pointing(lm, hand_label: str) -> bool:
    """Index extended; other fingers mostly down (tolerant for MediaPipe noise)."""
    thumb, index, middle, ring, pinky = _finger_states_static(lm, hand_label)
    curled = sum((not index, not middle, not ring, not pinky))
    return index and curled >= 2


def _hand_extended_count(lm, hand_label: str) -> int:
    return sum(_finger_states_static(lm, hand_label))


def _both_hands_open(lm1, lm2, label1: str, label2: str) -> bool:
    return _hand_extended_count(lm1, label1) >= 4 and _hand_extended_count(lm2, label2) >= 4


def _finger_states_static(lm, hand_label: str = "Right"):
    def dist(a, b):
        return ((lm[a].x - lm[b].x)**2 + (lm[a].y - lm[b].y)**2)**0.5

    # Rotation-invariant extension: tip is extended further than knuckle (MCP) from wrist (0)
    index = dist(8, 0) > dist(5, 0) * 1.12
    middle = dist(12, 0) > dist(9, 0) * 1.12
    ring = dist(16, 0) > dist(13, 0) * 1.12
    pinky = dist(20, 0) > dist(17, 0) * 1.12

    # Thumb extension: tip (4) is far from index MCP (5) compared to base (2)
    thumb = dist(4, 5) > dist(2, 5) * 1.12
    
    return thumb, index, middle, ring, pinky


def _palm_and_top_hand(lm1, lm2):
    """
    Thank / Again: lower hand = palm up, upper hand taps it.
    Works even when MediaPipe labels both hands as 'Five'.
    """
    if lm1[0].y >= lm2[0].y:
        palm_lm, top_lm = lm1, lm2
    else:
        palm_lm, top_lm = lm2, lm1

    if not _palms_facing_up(palm_lm):
        if _finger_spread(palm_lm) < 0.07:
            return None, None

    near_palm = (
        _tip_distance(top_lm, 8, palm_lm, 9) < 0.24
        or _tip_distance(top_lm, 0, palm_lm, 9) < 0.26
    )
    if not near_palm:
        return None, None
    if top_lm[0].y > palm_lm[0].y + 0.18:
        return None, None
    if _wrist_distance(palm_lm, top_lm) < 0.10:
        return None, None
    return palm_lm, top_lm


def _right_hand_taps_left_palm(palm_lm, other_lm) -> bool:
    """CSL Thank / Again: one palm up, other hand on/near that palm."""
    near = (
        _tip_distance(other_lm, 8, palm_lm, 9) < 0.24
        or _tip_distance(other_lm, 0, palm_lm, 9) < 0.26
    )
    if not near:
        return False
    if other_lm[0].y > palm_lm[0].y + 0.18:
        return False
    return 0.10 < _wrist_distance(palm_lm, other_lm) < 0.65


def _please_hands(lm1, lm2) -> bool:
    dist = _wrist_distance(lm1, lm2)
    if dist < 0.26 or dist > 0.68:
        return False
    return _palms_facing_up(lm1) and _palms_facing_up(lm2)


def _hand_on_chest(lm) -> bool:
    wx, wy = lm[0].x, lm[0].y
    # Restrict to chest-level vertically (wy >= 0.52) so it doesn't match face-level gestures
    if not (0.15 < wx < 0.85 and 0.52 < wy < 0.85):
        return False
    extended = sum(1 for i in (8, 12, 16, 20) if lm[i].y < lm[0].y + 0.02)
    return extended >= 3


class GestureEngine:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self._start_time = time.perf_counter()
        cfg = load_camera_config(camera_id)
        self._min_phrase_frames = int(cfg["min_phrase_frames"])
        self._min_hold_frames = int(cfg["min_hold_frames"])
        self._ml_confidence = float(cfg["ml_confidence"])
        self._mirror = bool(cfg["mirror"])
        self._use_ml = bool(cfg["use_ml"])
        self._classifier = GestureClassifier()

        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hand_landmarker.task')
        # Use CPU delegate to avoid Intel UHD EGL driver synchronization bugs on Linux
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU,
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        # Frame-skip state: run MediaPipe every 2nd frame for higher tracking speed
        self._frame_count = 0
        self._cached_hands_data: list[dict] = []

        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        self._again_trail: list[tuple[float, float]] = []
        self._phrase_stable: str | None = None
        self._phrase_candidate: str | None = None
        self._phrase_candidate_hits = 0
        self._phrase_clear_hits = 0
        self._simple_stable: str | None = None
        self._simple_candidate: str | None = None
        self._simple_candidate_hits = 0
        self._last_hands_data: list[dict] = []



        if self._classifier.enabled and self._use_ml:
            print(
                f"  [ml] Camera {camera_id}: classifier loaded "
                f"({len(self._classifier.labels)} labels, "
                f"confirm={self._min_phrase_frames} frames).",
                flush=True,
            )
        elif self._use_ml:
            print(
                f"  [ml] Camera {camera_id}: no model yet — using rules only. "
                f"Run scripts/collect_landmarks.py then scripts/train_classifier.py.",
                flush=True,
            )

    def _finger_states(self, lm, hand_label="Right"):
        return list(_finger_states_static(lm, hand_label))

    def _thumb_up_direction(self, lm):
        if lm[4].y < lm[0].y - 0.1:
            return 1
        if lm[4].y > lm[0].y + 0.1:
            return -1
        return 0

    def _ok_sign(self, lm):
        dist = ((lm[4].x - lm[8].x) ** 2 + (lm[4].y - lm[8].y) ** 2) ** 0.5
        return dist < 0.07

    def _thumbs_up_at_chest(self, lm, hand_label: str = "Right") -> bool:
        thumb, index, middle, ring, pinky = _finger_states_static(lm, hand_label)
        if not (thumb and not index and not middle and not ring and not pinky):
            return False
        if self._thumb_up_direction(lm) != 1:
            return False
        wx, wy = lm[0].x, lm[0].y
        # Lower half of screen (chest level) to distinguish from Drink
        return 0.15 < wx < 0.85 and 0.58 <= wy < 0.85

    def _drink_sign(self, lm, hand_label: str) -> bool:
        thumb, index, middle, ring, pinky = _finger_states_static(lm, hand_label)
        # Thumbs up handshape near mouth
        is_thumbs_up = thumb and (sum((not index, not middle, not ring, not pinky)) >= 3)
        if not is_thumbs_up:
            return False
        wx, wy = lm[0].x, lm[0].y
        # Upper/mid half of screen (mouth/face level)
        return 0.30 < wx < 0.70 and 0.35 < wy < 0.58

    def _delicious_sign(self, lm, hand_label: str) -> bool:
        # OK handshape near the mouth/cheek
        if not self._ok_sign(lm):
            return False
        wx, wy = lm[0].x, lm[0].y
        return 0.20 < wx < 0.80 and 0.35 < wy < 0.65

    def _like_sign(self, lm, hand_label: str) -> bool:
        # OK handshape at chest level
        if not self._ok_sign(lm):
            return False
        wx, wy = lm[0].x, lm[0].y
        return 0.58 <= wy < 0.85 and 0.30 < wx < 0.70

    def _i_me_sign(self, lm, hand_label: str) -> bool:
        # Flat hand resting on chest
        if _hand_extended_count(lm, hand_label) < 4:
            return False
        wx, wy = lm[0].x, lm[0].y
        return 0.58 <= wy < 0.85 and 0.30 < wx < 0.70

    def _you_sign(self, lm, hand_label: str) -> bool:
        # Index pointing forward at chest/conversational level
        if not _index_pointing(lm, hand_label):
            return False
        ix, iy = lm[8].x, lm[8].y
        return 0.55 <= iy < 0.80 and 0.30 < ix < 0.70

    def _age_sign(self, lm, hand_label: str) -> bool:
        # Flat hand near shoulder/chest level
        if _hand_extended_count(lm, hand_label) < 4:
            return False
        wx, wy = lm[0].x, lm[0].y
        return 0.45 <= wy < 0.70 and 0.20 < wx < 0.80

    def _beautiful_sign(self, lm, hand_label: str) -> bool:
        # Open hand (Five) next to the cheek/ear (side of face)
        if _hand_extended_count(lm, hand_label) < 4:
            return False
        wx, wy = lm[0].x, lm[0].y
        return 0.20 < wy < 0.60 and (wx < 0.38 or wx > 0.62)

    def _wrong_sign(self, lm, hand_label: str) -> bool:
        if not _index_pointing(lm, hand_label):
            return False
        # General pointing on screen (fallback check, evaluated last)
        ix, iy = lm[8].x, lm[8].y
        return iy < 0.70 and 0.10 < ix < 0.90

    def _understand_sign(self, lm, hand_label: str) -> bool:
        if not _index_pointing(lm, hand_label):
            return False
        # Index tip (8) next to the temple (high and to the side)
        ix, iy = lm[8].x, lm[8].y
        return iy < 0.42 and (ix < 0.45 or ix > 0.55)

    def _hearing_sign(self, lm, hand_label: str) -> bool:
        if not _index_pointing(lm, hand_label):
            return False
        # Index tip (8) next to the ear (outer sides of face)
        ix, iy = lm[8].x, lm[8].y
        return 0.20 < iy < 0.58 and (ix < 0.38 or ix > 0.62)

    def _deaf_sign(self, lm, hand_label: str) -> bool:
        if not _index_pointing(lm, hand_label):
            return False
        # Compound: index tip (8) near ear or near mouth
        ix, iy = lm[8].x, lm[8].y
        at_ear = 0.20 < iy < 0.58 and (ix < 0.38 or ix > 0.62)
        at_mouth = 0.38 < ix < 0.62 and 0.42 < iy < 0.60
        return at_ear or at_mouth

    def _detect_single_csl(self, hands_data):
        if len(hands_data) != 1:
            return None
        hand = hands_data[0]
        lm = hand["landmarks"]
        label = hand["label"]

        # 1. Height-specific thumbs up
        if self._drink_sign(lm, label):
            return "Drink"
        if self._thumbs_up_at_chest(lm, label):
            return "How Are You"

        # 2. OK shapes / Pinches
        if self._like_sign(lm, label):
            return "Like"
        if self._delicious_sign(lm, label):
            return "Delicious"

        # 3. Open hand shapes
        if self._i_me_sign(lm, label):
            return "I, me"
        if self._age_sign(lm, label):
            return "Age"
        if self._beautiful_sign(lm, label):
            return "Beautiful"

        # 4. Pointing gestures (most specific to least specific)
        if self._you_sign(lm, label):
            return "You"
        if self._understand_sign(lm, label):
            return "Understand"
        if self._hearing_sign(lm, label):
            return "Hearing"
        if self._deaf_sign(lm, label):
            return "Deaf"
        if self._wrong_sign(lm, label):  # Check general pointing last
            return "Wrong"
        return None



    def _detect_right_sign(self, lm1, lm2, label1, label2) -> bool:
        if not (_index_pointing(lm1, label1) and _index_pointing(lm2, label2)):
            return False
        if abs(lm1[8].x - lm2[8].x) > 0.20:
            return False
        top, bottom = (lm1, lm2) if lm1[8].y < lm2[8].y else (lm2, lm1)
        return top[8].y < bottom[8].y - 0.03

    def _detect_congratulation(self, lm1, lm2, label1: str, label2: str) -> bool:
        """Both hands high at shoulders, fingers spread (palms toward camera)."""
        if not _both_hands_open(lm1, lm2, label1, label2):
            return False
        avg_y = (lm1[0].y + lm2[0].y) / 2
        if avg_y > 0.58:
            return False
        if _finger_spread(lm1) < 0.08 or _finger_spread(lm2) < 0.08:
            return False
        return 0.28 < _wrist_distance(lm1, lm2) < 0.75

    def _detect_thank_or_again(self, palm_lm, other_lm) -> str | None:
        if not _right_hand_taps_left_palm(palm_lm, other_lm):
            self._again_trail.clear()
            return None

        self._again_trail.append(other_lm[0].x)
        if len(self._again_trail) > 10:
            self._again_trail.pop(0)

        if len(self._again_trail) >= 5:
            spread = max(self._again_trail) - min(self._again_trail)
            if spread > 0.04:
                return "Again"

        if other_lm[0].y < palm_lm[0].y - 0.12:
            return "Thank You"
        return "Thank You"

    def _detect_today(self, lm1, lm2, label1: str, label2: str) -> bool:
        g1 = self.detect_gesture(lm1, label1)
        g2 = self.detect_gesture(lm2, label2)
        if g1 == "Call Me" and g2 == "Call Me":
            avg_y = (lm1[0].y + lm2[0].y) / 2
            return 0.50 < avg_y < 0.85
        return False

    def _detect_people(self, lm1, lm2, label1: str, label2: str) -> bool:
        g1 = self.detect_gesture(lm1, label1)
        g2 = self.detect_gesture(lm2, label2)
        if g1 == "Two" and g2 == "Two":
            pointing_down1 = lm1[8].y > lm1[5].y and lm1[12].y > lm1[9].y
            pointing_down2 = lm2[8].y > lm2[5].y and lm2[12].y > lm2[9].y
            return pointing_down1 and pointing_down2
        return False

    def _detect_participate(self, lm1, lm2, label1: str, label2: str) -> bool:
        g1 = self.detect_gesture(lm1, label1)
        g2 = self.detect_gesture(lm2, label2)
        has_two = g1 == "Two" or g2 == "Two"
        has_open = _hand_extended_count(lm1, label1) >= 4 or _hand_extended_count(lm2, label2) >= 4
        if has_two and has_open:
            return _wrist_distance(lm1, lm2) < 0.24
        return False

    def _detect_happy(self, lm1, lm2, label1: str, label2: str) -> bool:
        if not _both_hands_open(lm1, lm2, label1, label2):
            return False
        avg_y = (lm1[0].y + lm2[0].y) / 2
        return 0.52 < avg_y < 0.82 and 0.18 < _wrist_distance(lm1, lm2) < 0.42

    def _detect_all_of_you(self, lm1, lm2, label1: str, label2: str) -> bool:
        """Both hands open, spread wide apart at chest level, pointing outward."""
        if not _both_hands_open(lm1, lm2, label1, label2):
            return False
        avg_y = (lm1[0].y + lm2[0].y) / 2
        dist = _wrist_distance(lm1, lm2)
        # Wider spread than Happy (>0.38) and hands are at chest/mid level
        return 0.45 < avg_y < 0.82 and dist > 0.38

    def _two_hand_gesture(self, hands_data):
        if len(hands_data) != 2:
            return None

        hand1, hand2 = hands_data
        lm1, lm2 = hand1["landmarks"], hand2["landmarks"]
        label1, label2 = hand1["label"], hand2["label"]
        dist = _wrist_distance(lm1, lm2)
        both_open = _both_hands_open(lm1, lm2, label1, label2)

        palm_lm, top_lm = _palm_and_top_hand(lm1, lm2)
        if palm_lm is not None:
            tap = self._detect_thank_or_again(palm_lm, top_lm)
            if tap:
                return tap

        if self._detect_right_sign(lm1, lm2, label1, label2):
            return "Right"

        if self._detect_congratulation(lm1, lm2, label1, label2):
            return "Congratulation"

        if self._detect_today(lm1, lm2, label1, label2):
            return "Today"

        if self._detect_people(lm1, lm2, label1, label2):
            return "People"

        if self._detect_participate(lm1, lm2, label1, label2):
            return "Participate"

        if self._detect_happy(lm1, lm2, label1, label2):
            return "Happy"

        if self._detect_all_of_you(lm1, lm2, label1, label2):
            return "All of you"

        if _index_pointing(lm1, label1) and _index_pointing(lm2, label2):
            return "Where From"

        if both_open:
            if dist < 0.22 and abs(lm1[0].y - lm2[0].y) < 0.15:
                return "Thank You"
            if _please_hands(lm1, lm2):
                return "Please"
            if abs(lm1[0].y - lm2[0].y) < 0.20 and 0.28 < dist < 0.62:
                return "Hello"

        gesture1 = self.detect_gesture(lm1, label1)
        gesture2 = self.detect_gesture(lm2, label2)
        if ((gesture1 == "Point Up" and gesture2 == "Point Up") or
            (gesture1 == "Point Up" and _is_open_palm(gesture2)) or
            (_is_open_palm(gesture1) and gesture2 == "Point Up")):
            return "Where From"

        return None

    def detect_gesture(self, lm, hand_label="Right"):
        if not lm:
            return "No Hand"

        f = self._finger_states(lm, hand_label)
        thumb, index, middle, ring, pinky = f
        count = sum(f)

        if count == 0:
            return "Fist"
        if count == 5:
            return "Five"
        if self._ok_sign(lm) and middle and ring and pinky:
            return "OK"
        if thumb and not index and not middle and not ring and not pinky:
            direction = self._thumb_up_direction(lm)
            if direction == 1:
                return "Thumbs Up"
            if direction == -1:
                return "Thumbs Down"
        if index and not middle and not ring and not pinky and not thumb:
            return "Point Up"
        if index and middle and not ring and not pinky and not thumb:
            return "Two"
        if index and middle and ring and not pinky and not thumb:
            return "Three"
        if index and middle and ring and pinky and not thumb:
            return "Four"
        if thumb and pinky and not index and not middle and not ring:
            return "Call Me"
        if thumb and index and not middle and not ring and not pinky:
            return "One"
        return "Open Hand"

    def _stabilize_label(
        self,
        candidate: str | None,
        *,
        stable: str | None,
        pending: str | None,
        pending_hits: int,
        clear_hits: int,
        min_confirm: int,
        min_hold: int,
    ) -> tuple[str | None, str | None, int, int]:
        """Require consecutive frames before switching; slow decay when hands drop."""
        if candidate is None:
            clear_hits += 1
            if clear_hits >= min_hold:
                return None, None, 0, clear_hits
            return stable, pending, pending_hits, clear_hits

        clear_hits = 0
        if candidate == pending:
            pending_hits += 1
        else:
            pending = candidate
            pending_hits = 1

        if pending_hits >= min_confirm:
            stable = candidate
        return stable, pending, pending_hits, clear_hits

    def _stabilize_phrase(self, candidate: str | None) -> str | None:
        self._phrase_stable, self._phrase_candidate, self._phrase_candidate_hits, self._phrase_clear_hits = (
            self._stabilize_label(
                candidate,
                stable=self._phrase_stable,
                pending=self._phrase_candidate,
                pending_hits=self._phrase_candidate_hits,
                clear_hits=self._phrase_clear_hits,
                min_confirm=self._min_phrase_frames,
                min_hold=self._min_hold_frames,
            )
        )
        return self._phrase_stable

    def _stabilize_simple(self, candidate: str | None) -> str | None:
        # Use the same frame threshold as phrases for consistent fast response
        min_confirm = max(2, self._min_phrase_frames // 2)
        self._simple_stable, self._simple_candidate, self._simple_candidate_hits, _ = (
            self._stabilize_label(
                candidate,
                stable=self._simple_stable,
                pending=self._simple_candidate,
                pending_hits=self._simple_candidate_hits,
                clear_hits=0,
                min_confirm=min_confirm,
                min_hold=2,
            )
        )
        return self._simple_stable

    def _detect_csl_rules(self, hands_data):
        """Rule-based CSL phrase detection (fallback when ML is unavailable)."""
        n = len(hands_data)

        if n == 2:
            return self._two_hand_gesture(hands_data)
        if n == 1:
            # Check single CSL rules first (Understand, Hearing, Drink, etc.)
            res = self._detect_single_csl(hands_data)
            if res is not None:
                return res
            return None
        if n >= 2:
            return self._two_hand_gesture(hands_data[:2])
        return None

    def _detect_csl_phrase(self, hands_data):
        """ML classifier first, then rule engine; temporal smoothing on output."""
        candidate = None

        if self._use_ml and self._classifier.enabled and hands_data:
            from src.landmark_features import hands_to_feature_vector
            feat = hands_to_feature_vector(hands_data, mirror=self._mirror)
            if feat is not None:
                label, confidence = self._classifier.predict(feat)
                if label and label != "No Hand" and confidence >= self._ml_confidence:
                    candidate = label

        if candidate is None:
            candidate = self._detect_csl_rules(hands_data)

        return self._stabilize_phrase(candidate)

    @property
    def last_hands_data(self) -> list[dict]:
        return self._last_hands_data


    def _run_detection(self, frame, rgb_frame):
        """Run MediaPipe on an RGB frame; return hand landmark dicts."""
        timestamp_ms = int((time.perf_counter() - self._start_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        try:
            results = self.detector.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"MediaPipe detection error: {e}")
            return []

        hands_data = []
        if not results.hand_landmarks:
            return hands_data

        h, w, _ = frame.shape
        for i, hand_lm in enumerate(results.hand_landmarks):
            hand_info = results.handedness[i][0]
            # Draw connecting lines only to reduce CPU drawing overhead
            for connection in self.HAND_CONNECTIONS:
                pt1 = hand_lm[connection[0]]
                pt2 = hand_lm[connection[1]]
                x1, y1 = int(pt1.x * w), int(pt1.y * h)
                x2, y2 = int(pt2.x * w), int(pt2.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            hands_data.append({"landmarks": hand_lm, "label": hand_info.category_name})
        return hands_data

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        # ── Speed optimisation 1: shrink to max 256px before TFLite inference ──
        max_side = 256
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            small = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            small = frame

        # ── Speed optimisation 2: run MediaPipe every 2nd frame ─────────────────
        self._frame_count += 1
        if self._frame_count % 2 == 1:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            self._cached_hands_data = self._run_detection(frame, rgb)
        hands_data = self._cached_hands_data
        self._last_hands_data = hands_data

        gestures = []
        if hands_data:
            phrase = self._detect_csl_phrase(hands_data)
            if phrase:
                gestures = [phrase]
            elif len(hands_data) == 2:
                gestures = []
            else:
                for hand in hands_data:
                    raw = self.detect_gesture(hand['landmarks'], hand['label'])
                    stable = self._stabilize_simple(raw)
                    if stable:
                        gestures.append(stable)

        return frame, gestures

    def release(self):
        self.detector.close()
