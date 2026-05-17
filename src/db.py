"""PostgreSQL logging for gesture detection."""

from __future__ import annotations

import os
import threading
from typing import Optional

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:  # pragma: no cover
    psycopg2 = None
    RealDictCursor = None

TWO_HAND_GESTURES = frozenset({
    "Hello", "How Are You", "Where From",
    "Thank You", "Please", "Sorry",
    "Right", "Again", "Congratulation",
})


def _build_database_url() -> Optional[str]:
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")

    if not name:
        return None

    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


class GestureDatabase:
    """Thread-safe PostgreSQL helper for sessions and gesture events."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or _build_database_url()
        self._conn = None
        self._lock = threading.Lock()
        self.session_id: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.database_url)

    def connect(self) -> None:
        if not self.enabled:
            raise RuntimeError(
                "Database not configured. Set DATABASE_URL or DB_NAME in .env"
            )
        if psycopg2 is None:
            raise RuntimeError(
                "psycopg2 is not installed. Run: pip install psycopg2-binary"
            )
        if self._conn is not None and not self._conn.closed:
            return

        self._conn = psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
        self._conn.autocommit = False

    def test_connection(self) -> dict:
        self.connect()
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute("SELECT current_database() AS db, NOW() AS server_time")
                row = dict(cur.fetchone())
                cur.execute("SELECT COUNT(*) AS n FROM gestures")
                row["gesture_count"] = cur.fetchone()["n"]
        return row

    def start_session(self, mode: str = "live", camera_count: int = 1) -> str:
        self.connect()
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (mode, camera_count)
                    VALUES (%s, %s)
                    RETURNING id
                    """,
                    (mode, camera_count),
                )
                self.session_id = str(cur.fetchone()["id"])
            self._conn.commit()
        return self.session_id

    def end_session(self) -> None:
        if not self.session_id or self._conn is None:
            return
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET ended_at = NOW() WHERE id = %s",
                    (self.session_id,),
                )
            self._conn.commit()
        self.session_id = None

    def log_gesture(
        self,
        gesture_name: str,
        camera_id: int = 0,
        *,
        session_id: Optional[str] = None,
    ) -> bool:
        """Insert one gesture event. Returns False if gesture name not in DB."""
        sid = session_id or self.session_id
        if not sid:
            return False

        hands_count = 2 if gesture_name in TWO_HAND_GESTURES else 1

        self.connect()
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO gesture_events
                        (gesture_id, session_id, camera_id, hands_count)
                    SELECT g.id, %s::uuid, %s, %s
                    FROM gestures g
                    WHERE g.name_en = %s
                    RETURNING id
                    """,
                    (sid, camera_id, hands_count, gesture_name),
                )
                inserted = cur.fetchone()
            if inserted:
                self._conn.commit()
                return True
            self._conn.rollback()
            return False

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None
