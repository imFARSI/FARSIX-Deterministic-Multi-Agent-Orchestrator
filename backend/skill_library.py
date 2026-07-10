"""
FARSIX Skill Library — SQLite-backed registry of reusable mission skills.

Every successfully completed mission is distilled into a "skill" and saved here.
New missions query the library first; if a matching skill exists its context is
pre-loaded into the agent prompts (like few-shot conditioning).

Schema:
    skills(id, name, input_type, description, context_snapshot,
           usage_count, success_rate, created_at, last_used_at)
"""

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional


DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")
DB_PATH = os.path.join(DB_DIR, "missions.db")

os.makedirs(DB_DIR, exist_ok=True)


@dataclass
class Skill:
    id: int
    name: str
    input_type: str          # "text" | "pdf" | "csv" | "image"
    description: str
    context_snapshot: str    # JSON-encoded summary of key findings
    usage_count: int
    success_rate: float      # 0.0 – 1.0
    created_at: float
    last_used_at: float


@dataclass
class NewSkillPayload:
    """What the mission engine passes when saving a skill."""
    name: str
    input_type: str
    description: str
    context_snapshot: str    # JSON string or plain text


class SkillLibrary:
    """
    Thread-safe SQLite skill registry.

    Usage:
        lib = SkillLibrary()
        lib.save_skill(NewSkillPayload(...))
        skills = lib.search_skills("factory inspection", input_type="text")
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------ #
    #  Schema                                                               #
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    name            TEXT NOT NULL,
                    input_type      TEXT NOT NULL,
                    description     TEXT NOT NULL,
                    context_snapshot TEXT NOT NULL,
                    usage_count     INTEGER DEFAULT 0,
                    success_rate    REAL DEFAULT 1.0,
                    created_at      REAL NOT NULL,
                    last_used_at    REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS missions (
                    id          TEXT PRIMARY KEY,
                    goal        TEXT NOT NULL,
                    input_type  TEXT NOT NULL,
                    state       TEXT NOT NULL,
                    result      TEXT,
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL,
                    retry_count INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ #
    #  Skills CRUD                                                          #
    # ------------------------------------------------------------------ #

    def save_skill(self, payload: NewSkillPayload) -> Skill:
        """Insert a new skill and return the created Skill object."""
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO skills
                        (name, input_type, description, context_snapshot,
                         usage_count, success_rate, created_at, last_used_at)
                    VALUES (?, ?, ?, ?, 0, 1.0, ?, ?)
                    """,
                    (payload.name, payload.input_type, payload.description,
                     payload.context_snapshot, now, now),
                )
                conn.commit()
                skill_id = cur.lastrowid

        return self.get_skill_by_id(skill_id)

    def get_skill_by_id(self, skill_id: int) -> Optional[Skill]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM skills WHERE id = ?", (skill_id,)
            ).fetchone()
        return self._row_to_skill(row) if row else None

    def get_all_skills(self) -> List[Skill]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM skills ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def search_skills(self, query: str, input_type: Optional[str] = None,
                      limit: int = 5) -> List[Skill]:
        """
        Simple keyword search over name + description.
        Returns skills ordered by usage_count DESC.
        """
        like_q = f"%{query}%"
        with self._connect() as conn:
            if input_type:
                rows = conn.execute(
                    """
                    SELECT * FROM skills
                    WHERE (name LIKE ? OR description LIKE ?)
                      AND input_type = ?
                    ORDER BY usage_count DESC LIMIT ?
                    """,
                    (like_q, like_q, input_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM skills
                    WHERE name LIKE ? OR description LIKE ?
                    ORDER BY usage_count DESC LIMIT ?
                    """,
                    (like_q, like_q, limit),
                ).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def increment_usage(self, skill_id: int, success: bool = True) -> None:
        """Update usage_count and rolling success_rate for a skill."""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT usage_count, success_rate FROM skills WHERE id = ?",
                    (skill_id,),
                ).fetchone()
                if not row:
                    return
                new_count = row["usage_count"] + 1
                # Exponential moving average for success rate
                alpha = 0.2
                new_rate = (1 - alpha) * row["success_rate"] + alpha * (1.0 if success else 0.0)
                conn.execute(
                    """
                    UPDATE skills
                    SET usage_count = ?, success_rate = ?, last_used_at = ?
                    WHERE id = ?
                    """,
                    (new_count, new_rate, time.time(), skill_id),
                )
                conn.commit()

    def delete_skill(self, skill_id: int) -> bool:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
                conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------ #
    #  Mission persistence                                                  #
    # ------------------------------------------------------------------ #

    def save_mission(self, mission_id: str, goal: str, input_type: str,
                     state: str, result: Optional[str] = None,
                     retry_count: int = 0) -> None:
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO missions
                        (id, goal, input_type, state, result, created_at, updated_at, retry_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        state      = excluded.state,
                        result     = excluded.result,
                        updated_at = excluded.updated_at,
                        retry_count = excluded.retry_count
                    """,
                    (mission_id, goal, input_type, state, result, now, now, retry_count),
                )
                conn.commit()

    def get_mission(self, mission_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM missions WHERE id = ?", (mission_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_all_missions(self) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM missions ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    #  Internal                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _row_to_skill(row: sqlite3.Row) -> Skill:
        return Skill(
            id=row["id"],
            name=row["name"],
            input_type=row["input_type"],
            description=row["description"],
            context_snapshot=row["context_snapshot"],
            usage_count=row["usage_count"],
            success_rate=row["success_rate"],
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
        )

    def skill_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM skills").fetchone()
        return row["cnt"] if row else 0


# Singleton
_skill_library: Optional[SkillLibrary] = None


def get_skill_library() -> SkillLibrary:
    global _skill_library
    if _skill_library is None:
        _skill_library = SkillLibrary()
    return _skill_library
