#!/usr/bin/env python3
"""
Demo script to test Khmer gesture recognition
Run this to see how gestures are detected without needing a camera
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.gesture_engine import GESTURE_KHMER

def demo_gestures():
    """Display all available gestures and their Khmer translations."""
    print("🎭 Khmer Gesture Recognition - Available Gestures")
    print("=" * 60)

    print("\n🤏 Single Hand Gestures:")
    single_hand = [
        "Fist", "Open Hand", "Peace", "Thumbs Up", "Thumbs Down",
        "One", "Three", "Four", "OK", "Rock", "Call Me", "Point Up"
    ]

    for gesture in single_hand:
        if gesture in GESTURE_KHMER:
            khmer, english = GESTURE_KHMER[gesture]
            print(f"  {gesture:12} → {khmer} ({english})")

    print("\n🤝 Two-Hand Phrase Gestures:")
    two_hand = [
        "Hello", "How Are You", "Where From", "Thank You", "Please", "Sorry"
    ]

    for gesture in two_hand:
        if gesture in GESTURE_KHMER:
            khmer, english = GESTURE_KHMER[gesture]
            print(f"  {gesture:12} → {khmer} ({english})")

    print("\n📖 How to use:")
    print("  1. Run: python3 main.py")
    print("  2. Make gestures with one or both hands")
    print("  3. See Khmer text and English translation on screen")
    print("  4. Press 'q' or Ctrl+C to exit")

    print("\n✋ Two-hand gesture instructions:")
    instructions = {
        "Hello": "Hold both hands open, palms facing each other",
        "How Are You": "One hand open palm up, other pointing up",
        "Where From": "Both hands pointing or one pointing + one open",
        "Thank You": "Bring both open hands together (prayer position)",
        "Please": "Hold both hands open with palms up",
        "Sorry": "Cross open hands over your chest"
    }

    for gesture, instruction in instructions.items():
        print(f"  • {gesture}: {instruction}")

if __name__ == "__main__":
    demo_gestures()