"""User conversation history manager using SQLite.

Stores advisor Q&A pairs keyed by a stable user_id and an auto-generated session_id.
Lightweight and file-based; fine for Streamlit deployments.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
import json
import time


@dataclass
class QA:
    session_id: str
    role: str  # 'user' | 'assistant'
    content: str
    ts: float
    meta: Optional[Dict[str, Any]] = None


class HistoryManager:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or Path(__file__).parent.parent / "data" / "user_history.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL
                );
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    ts REAL NOT NULL,
                    meta TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );
                """
            )
            conn.commit()

    def ensure_user(self, user_id: str) -> None:
        now = time.time()
        with self._connect() as conn:
            c = conn.cursor()
            c.execute("INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?,?)", (user_id, now))
            conn.commit()

    def create_session(self, user_id: str) -> str:
        self.ensure_user(user_id)
        session_id = str(uuid.uuid4())
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO sessions(session_id, user_id, started_at) VALUES (?,?,?)",
                (session_id, user_id, time.time()),
            )
            conn.commit()
        return session_id

    def log_qa(self, user_id: str, question: str, answer: str, *, session_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> str:
        """Log a Q/A pair and return the session_id used."""
        if not session_id:
            session_id = self.create_session(user_id)
        self.ensure_user(user_id)
        now = time.time()
        with self._connect() as conn:
            c = conn.cursor()
            # user question
            c.execute(
                "INSERT INTO messages(session_id, role, content, ts, meta) VALUES (?,?,?,?,?)",
                (session_id, "user", question, now, None),
            )
            # assistant answer
            c.execute(
                "INSERT INTO messages(session_id, role, content, ts, meta) VALUES (?,?,?,?,?)",
                (session_id, "assistant", answer, now + 0.001, json.dumps(meta or {})),
            )
            conn.commit()
        return session_id

    def get_recent_qa(self, user_id: str, limit: int = 50) -> List[QA]:
        with self._connect() as conn:
            c = conn.cursor()
            # get sessions for user (latest first)
            c.execute(
                "SELECT session_id FROM sessions WHERE user_id=? ORDER BY started_at DESC",
                (user_id,),
            )
            session_ids = [row[0] for row in c.fetchall()]
            if not session_ids:
                return []
            # fetch messages for sessions (limit roughly by messages)
            q_marks = ",".join(["?"] * len(session_ids))
            c.execute(
                f"SELECT session_id, role, content, ts, meta FROM messages WHERE session_id IN ({q_marks}) ORDER BY ts DESC LIMIT ?",
                (*session_ids, limit),
            )
            rows = c.fetchall()
            out: List[QA] = []
            for s_id, role, content, ts, meta in rows:
                md = json.loads(meta) if meta else None
                out.append(QA(session_id=s_id, role=role, content=content, ts=ts, meta=md))
            # return chronological
            return list(reversed(out))

    def clear_user(self, user_id: str) -> None:
        """Delete all sessions and messages for a user (keeps user row)."""
        with self._connect() as conn:
            c = conn.cursor()
            c.execute("SELECT session_id FROM sessions WHERE user_id=?", (user_id,))
            sids = [row[0] for row in c.fetchall()]
            if sids:
                q_marks = ",".join(["?"] * len(sids))
                c.execute(f"DELETE FROM messages WHERE session_id IN ({q_marks})", (*sids,))
                c.execute(f"DELETE FROM sessions WHERE session_id IN ({q_marks})", (*sids,))
            conn.commit()
