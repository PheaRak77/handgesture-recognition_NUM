"""Sync GESTURE_KHMER from code into PostgreSQL. Run: python scripts/sync_gestures.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from src.db import GestureDatabase
from src.gesture_engine import GESTURE_KHMER

TWO_HAND = frozenset({
    "Hello", "How Are You", "Where From",
    "Thank You", "Please", "Sorry",
    "Right", "Again", "Congratulation",
})

REMOVED = frozenset({"Rock", "Peace"})


def main():
    db = GestureDatabase()
    if not db.enabled:
        print("Configure .env first (copy env.example).")
        sys.exit(1)

    db.connect()
    upserted = 0
    with db._lock:
        with db._conn.cursor() as cur:
            for name_en, (text_khmer, text_english) in GESTURE_KHMER.items():
                gtype = "two_hand" if name_en in TWO_HAND else "single_hand"
                cur.execute(
                    """
                    INSERT INTO gestures (name_en, text_khmer, text_english, gesture_type, is_active)
                    VALUES (%s, %s, %s, %s, TRUE)
                    ON CONFLICT (name_en) DO UPDATE SET
                        text_khmer   = EXCLUDED.text_khmer,
                        text_english = EXCLUDED.text_english,
                        gesture_type = EXCLUDED.gesture_type,
                        is_active    = TRUE,
                        updated_at   = NOW()
                    """,
                    (name_en, text_khmer, text_english, gtype),
                )
                upserted += 1

            for name in REMOVED:
                cur.execute(
                    "UPDATE gestures SET is_active = FALSE, updated_at = NOW() WHERE name_en = %s",
                    (name,),
                )

            cur.execute("SELECT COUNT(*) AS n FROM gestures WHERE is_active = TRUE")
            active = cur.fetchone()["n"]
        db._conn.commit()

    print(f"Synced {upserted} gestures from GESTURE_KHMER.")
    print(f"Active gestures in database: {active}")
    db.close()


if __name__ == "__main__":
    main()
