import cv2
import mediapipe as mp

# ─────────────────────────────────────────────────────────────────────────────
# Khmer meanings for every recognised gesture
# ─────────────────────────────────────────────────────────────────────────────
GESTURE_KHMER = {
    "No Hand":        ("គ្មានដៃ",        "No hand detected"),
    "Fist":           ("ក្ដាប់ដៃ",       "Fist / Stop"),
    "Open Hand":      ("ដៃបើក",         "Open hand / Hello"),
    "Peace":          ("សន្តិភាព",       "Peace / Victory (2 fingers)"),
    "Thumbs Up":      ("ល្អណាស់",        "Thumbs Up / Good"),
    "Thumbs Down":    ("មិនល្អ",         "Thumbs Down / Bad"),
    "One":            ("មួយ",           "Number One / Index finger"),
    "Three":          ("បី",            "Number Three"),
    "Four":           ("បួន",           "Number Four"),
    "OK":             ("យល់ព្រម",        "OK / Agree"),
    "Rock":           ("រ៉ុក",           "Rock / Metal sign"),
    "Call Me":        ("ហៅខ្ញុំ",        "Call me / Phone"),
    "Point Up":       ("ចង្អុលឡើង",      "Point Up / Attention"),

    # ── Two-hand phrase gestures ───────────────────────────────────────────
    "Hello":          ("សួស្តី",         "Hello / Greeting"),
    "How Are You":    ("សុខសប្បាយជា",   "How are you?"),
    "Where From":     ("មកពីណា",        "Where are you from?"),
    "Thank You":      ("អរគុណ",         "Thank you"),
    "Please":         ("សូម",           "Please"),
    "Sorry":          ("សុំទោស",        "Sorry"),
}


class GestureEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw  = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    # ── Landmark indices (MediaPipe 21-point hand model) ─────────────────────
    # Fingertips : 4(thumb) 8(index) 12(middle) 16(ring) 20(pinky)
    # PIP joints : 6(index) 10(middle) 14(ring) 18(pinky)
    # MCP joints : 5(index) 9(middle) 13(ring) 17(pinky)
    # Wrist      : 0

    def _finger_states(self, lm, hand_label="Right"):
        """Return [thumb, index, middle, ring, pinky] — True = extended."""
        # Four fingers: tip above (lower y) than PIP joint
        index  = lm[8].y  < lm[6].y
        middle = lm[12].y < lm[10].y
        ring   = lm[16].y < lm[14].y
        pinky  = lm[20].y < lm[18].y

        # Thumb: compare tip x vs MCP x (mirrors for left/right hand)
        if hand_label == "Right":
            thumb = lm[4].x < lm[3].x   # tip to the left of IP joint
        else:
            thumb = lm[4].x > lm[3].x

        return [thumb, index, middle, ring, pinky]

    def _thumb_up_direction(self, lm):
        """Return +1 if thumb points UP, -1 if DOWN, 0 if sideways."""
        # Compare thumb tip y vs wrist y
        if lm[4].y < lm[0].y - 0.1:
            return 1    # UP
        if lm[4].y > lm[0].y + 0.1:
            return -1   # DOWN
        return 0

    def _ok_sign(self, lm):
        """Detect OK: thumb tip close to index tip, other fingers open."""
        dist = ((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2) ** 0.5
        return dist < 0.07

    def _two_hand_gesture(self, hands_data):
        """Detect two-hand gestures for phrases."""
        if len(hands_data) != 2:
            return None

        hand1, hand2 = hands_data
        gesture1 = self.detect_gesture(hand1['landmarks'], hand1['label'])
        gesture2 = self.detect_gesture(hand2['landmarks'], hand2['label'])

        # ── Hello: Two open hands facing each other ────────────────────────
        if (gesture1 == "Open Hand" and gesture2 == "Open Hand"):
            # Check if hands are facing each other (palms inward)
            lm1, lm2 = hand1['landmarks'], hand2['landmarks']
            # Simple check: hands at similar height and facing inward
            if abs(lm1[0].y - lm2[0].y) < 0.2:  # Similar vertical position
                return "Hello"

        # ── How Are You: One open hand palm up, one pointing/questioning ───
        if ((gesture1 == "Open Hand" and gesture2 == "Point Up") or
            (gesture1 == "Point Up" and gesture2 == "Open Hand")):
            return "How Are You"

        # ── Where From: Both hands pointing or one pointing, one open ─────
        if ((gesture1 == "Point Up" and gesture2 == "Point Up") or
            (gesture1 == "Point Up" and gesture2 == "Open Hand") or
            (gesture1 == "Open Hand" and gesture2 == "Point Up")):
            return "Where From"

        # ── Thank You: Hands together (prayer position) ────────────────────
        if (gesture1 == "Open Hand" and gesture2 == "Open Hand"):
            lm1, lm2 = hand1['landmarks'], hand2['landmarks']
            # Check if hands are close together (prayer position)
            dist = ((lm1[0].x - lm2[0].x)**2 + (lm1[0].y - lm2[0].y)**2) ** 0.5
            if dist < 0.3:  # Hands close together
                return "Thank You"

        # ── Please: Two open hands with palms up ───────────────────────────
        if (gesture1 == "Open Hand" and gesture2 == "Open Hand"):
            # This is a fallback - in practice we'd need more sophisticated
            # palm orientation detection
            return "Please"

        # ── Sorry: Hands crossed over chest ────────────────────────────────
        if (gesture1 == "Open Hand" and gesture2 == "Open Hand"):
            lm1, lm2 = hand1['landmarks'], hand2['landmarks']
            # Check if hands are crossed (one hand over the other)
            if abs(lm1[0].x - lm2[0].x) < 0.2 and abs(lm1[0].y - lm2[0].y) > 0.1:
                return "Sorry"

        return None

    def detect_gesture(self, lm, hand_label="Right"):
        """Classify a single hand into one of the known gesture names."""
        if not lm:
            return "No Hand"

        f = self._finger_states(lm, hand_label)   # [thumb, idx, mid, ring, pinky]
        thumb, index, middle, ring, pinky = f
        count = sum(f)

        # ── No fingers up → Fist ────────────────────────────────────────────
        if count == 0:
            return "Fist"

        # ── All 5 up → Open Hand ────────────────────────────────────────────
        if count == 5:
            return "Open Hand"

        # ── OK sign: thumb & index tips touching ────────────────────────────
        if self._ok_sign(lm) and middle and ring and pinky:
            return "OK"

        # ── Thumb only ──────────────────────────────────────────────────────
        if thumb and not index and not middle and not ring and not pinky:
            direction = self._thumb_up_direction(lm)
            if direction == 1:
                return "Thumbs Up"
            if direction == -1:
                return "Thumbs Down"

        # ── Index only → Point Up ───────────────────────────────────────────
        if index and not middle and not ring and not pinky and not thumb:
            return "Point Up"

        # ── Index + Middle → Peace ──────────────────────────────────────────
        if index and middle and not ring and not pinky:
            return "Peace"

        # ── Index + Middle + Ring → Three ───────────────────────────────────
        if index and middle and ring and not pinky and not thumb:
            return "Three"

        # ── Index + Middle + Ring + Pinky (no thumb) → Four ─────────────────
        if index and middle and ring and pinky and not thumb:
            return "Four"

        # ── Index + Pinky (rock/metal) ───────────────────────────────────────
        if index and pinky and not middle and not ring:
            return "Rock"

        # ── Thumb + Pinky (call me) ──────────────────────────────────────────
        if thumb and pinky and not index and not middle and not ring:
            return "Call Me"

        # ── Thumb + Index (one / gun) ────────────────────────────────────────
        if thumb and index and not middle and not ring and not pinky:
            return "One"

        # Default fallback
        return "Open Hand"

    # ── Main processing entry-point ──────────────────────────────────────────
    def process_frame(self, frame):
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        gestures = []
        hands_data = []

        if results.multi_hand_landmarks:
            # Collect data for all detected hands
            for hand_lm, hand_info in zip(
                results.multi_hand_landmarks,
                results.multi_handedness,
            ):
                hand_label = hand_info.classification[0].label  # "Left" / "Right"
                self.mp_draw.draw_landmarks(
                    frame, hand_lm, self.mp_hands.HAND_CONNECTIONS
                )

                hand_data = {
                    'landmarks': hand_lm.landmark,
                    'label': hand_label
                }
                hands_data.append(hand_data)

                # Still detect single-hand gestures
                gesture = self.detect_gesture(hand_lm.landmark, hand_label)
                gestures.append(gesture)

            # Check for two-hand gestures (phrases)
            two_hand_gesture = self._two_hand_gesture(hands_data)
            if two_hand_gesture:
                # If we detect a two-hand gesture, prioritize it
                gestures = [two_hand_gesture]

        return frame, gestures

    def release(self):
        self.hands.close()