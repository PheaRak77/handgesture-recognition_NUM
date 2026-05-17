"""Test PostgreSQL connection. Run: python scripts/test_db.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from src.db import GestureDatabase


def main():
    db = GestureDatabase()
    if not db.enabled:
        print("Database not configured.")
        print("Copy .env.example to .env and set DB_NAME / DB_PASSWORD.")
        sys.exit(1)

    try:
        info = db.test_connection()
        print("Connected successfully.")
        print(f"  database      : {info['db']}")
        print(f"  server_time   : {info['server_time']}")
        print(f"  gesture_count : {info['gesture_count']}")
        if info["gesture_count"] == 0:
            print("\nRun database/02_seed_gestures.sql in pgAdmin to insert gestures.")
    except Exception as exc:
        print(f"Connection failed: {exc}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
