import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
    "Sorry":          ("សុំទោស",        "Sorry"),
    "Right":          ("ត្រូវ",          "Right"),
    "Wrong":          ("ខុស",            "Wrong"),
    "Understand":     ("យល់",            "Understand"),
    "Again":          ("ម្តងទៀត",        "Again"),
    "Deaf":           ("ថ្លង់",           "Deaf"),
    "Congratulation": ("អបអរសាទរ",       "Congratulation"),
    "Hearing":        ("ស្តាប់ឮ",         "Hearing"),
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
    index = lm[8].y < lm[6].y
    middle = lm[12].y < lm[10].y
    ring = lm[16].y < lm[14].y
    pinky = lm[20].y < lm[18].y
    if hand_label == "Right":
        thumb = lm[4].x < lm[3].x
    else:
        thumb = lm[4].x > lm[3].x
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
    if not (0.28 < wx < 0.72 and 0.30 < wy < 0.72):
        return False
    extended = sum(1 for i in (8, 12, 16, 20) if lm[i].y < lm[0].y + 0.02)
    return extended >= 3


class GestureEngine:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        self._sorry_trail: list[tuple[float, float]] = []
        self._again_trail: list[tuple[float, float]] = []
        self._phrase_stable: str | None = None
        self._phrase_hits = 0

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
        return 0.30 < wx < 0.70 and 0.38 < wy < 0.78

    def _wrong_sign(self, lm, hand_label: str) -> bool:
        if not _index_only(lm, hand_label):
            return False
        wx, wy, iy = lm[0].x, lm[0].y, lm[8].y
        return iy < 0.48 and 0.22 < wx < 0.78

    def _understand_sign(self, lm, hand_label: str) -> bool:
        if not _index_only(lm, hand_label):
            return False
        wx, wy = lm[0].x, lm[0].y
        at_temple = wy < 0.42 and (wx < 0.38 or wx > 0.62)
        return at_temple

    def _hearing_sign(self, lm, hand_label: str) -> bool:
        if not _index_only(lm, hand_label):
            return False
        wx, wy = lm[0].x, lm[0].y
        return 0.26 < wy < 0.58 and (wx < 0.34 or wx > 0.66)

    def _deaf_sign(self, lm, hand_label: str) -> bool:
        if not _index_only(lm, hand_label):
            return False
        wx, wy = lm[0].x, lm[0].y
        at_ear = wy < 0.50 and (wx < 0.32 or wx > 0.68)
        at_mouth = 0.38 < wx < 0.62 and 0.52 < wy < 0.78
        return at_ear or at_mouth

    def _detect_single_csl(self, hands_data):
        if len(hands_data) != 1:
            return None
        hand = hands_data[0]
        lm = hand["landmarks"]
        label = hand["label"]

        if self._thumbs_up_at_chest(lm, label):
            return "How Are You"
        if self._wrong_sign(lm, label):
            return "Wrong"
        if self._understand_sign(lm, label):
            return "Understand"
        if self._hearing_sign(lm, label):
            return "Hearing"
        if self._deaf_sign(lm, label):
            return "Deaf"
        return None

    def _detect_sorry(self, hands_data) -> bool:
        """Sorry uses one hand on chest — skip when clearly doing two-hand signs."""
        if len(hands_data) == 2:
            lm1 = hands_data[0]["landmarks"]
            lm2 = hands_data[1]["landmarks"]
            if _both_hands_open(
                lm1, lm2, hands_data[0]["label"], hands_data[1]["label"]
            ):
                return False
            if _wrist_distance(lm1, lm2) < 0.55:
                on_chest = sum(1 for h in hands_data if _hand_on_chest(h["landmarks"]))
                if on_chest < 2:
                    pass
                else:
                    return False

        candidates = []
        for hand in hands_data:
            lm = hand["landmarks"]
            if _hand_on_chest(lm):
                candidates.append((lm[0].x, lm[0].y))

        if not candidates:
            self._sorry_trail.clear()
            return False

        if len(hands_data) == 2 and len(candidates) == 1:
            chest_hand = next(
                h for h in hands_data if _hand_on_chest(h["landmarks"])
            )
            other = next(
                h for h in hands_data if h is not chest_hand
            )
            if _hand_extended_count(other["landmarks"], other["label"]) >= 4:
                return False
            if _wrist_distance(chest_hand["landmarks"], other["landmarks"]) > 0.40:
                return False

        cx, cy = candidates[0]
        self._sorry_trail.append((cx, cy))
        if len(self._sorry_trail) > 12:
            self._sorry_trail.pop(0)
        return True

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

    def _stabilize_phrase(self, candidate: str | None) -> str | None:
        """Hold phrase for a few frames so two-hand signs don't flicker away."""
        if candidate is None:
            self._phrase_hits = max(0, self._phrase_hits - 1)
            if self._phrase_hits <= 0:
                self._phrase_stable = None
            return self._phrase_stable

        if candidate == self._phrase_stable:
            self._phrase_hits = min(self._phrase_hits + 1, 8)
            return candidate

        self._phrase_hits += 1
        if self._phrase_hits >= 2 or self._phrase_stable is None:
            self._phrase_stable = candidate
            self._phrase_hits = 2
            return candidate
        return self._phrase_stable

    def _detect_csl_phrase(self, hands_data):
        """CSL phrase: two hands → two-hand signs first; one hand → single-hand signs."""
        n = len(hands_data)
        candidate = None

        if n == 2:
            candidate = self._two_hand_gesture(hands_data)
        elif n == 1:
            if self._detect_sorry(hands_data):
                candidate = "Sorry"
            else:
                candidate = self._detect_single_csl(hands_data)
        elif n >= 2:
            candidate = self._two_hand_gesture(hands_data[:2])

        return self._stabilize_phrase(candidate)

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        max_side = 640
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            small = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            results = self.detector.detect(mp_image)
        except Exception as e:
            print(f"MediaPipe detection error: {e}")
            return frame, []

        gestures = []
        hands_data = []

        if results.hand_landmarks:
            h, w, _ = frame.shape
            for i, hand_lm in enumerate(results.hand_landmarks):
                hand_info = results.handedness[i][0]
                hand_label = hand_info.category_name

                for connection in self.HAND_CONNECTIONS:
                    pt1 = hand_lm[connection[0]]
                    pt2 = hand_lm[connection[1]]
                    x1, y1 = int(pt1.x * w), int(pt1.y * h)
                    x2, y2 = int(pt2.x * w), int(pt2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for lm in hand_lm:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

                hands_data.append({
                    'landmarks': hand_lm,
                    'label': hand_label,
                })

            phrase = self._detect_csl_phrase(hands_data)
            if phrase:
                gestures = [phrase]
            elif len(hands_data) == 2:
                gestures = []
            else:
                for hand in hands_data:
                    gestures.append(
                        self.detect_gesture(hand['landmarks'], hand['label'])
                    )

        return frame, gestures

    def release(self):
        self.detector.close()
