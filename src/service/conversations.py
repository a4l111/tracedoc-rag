"""Shared LAN conversation history; prior turns are context, never retrieval evidence."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


MAX_HISTORY_TURNS = 20
MAX_HISTORY_CHARS = 48_000


class Conversations:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY, title TEXT, updated REAL)")
            db.execute("CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                       "conversation_id TEXT, question TEXT, final_answer TEXT, snapshot_id TEXT, "
                       "evidence_json TEXT, created REAL)")
            db.execute("CREATE INDEX IF NOT EXISTS turns_conversation ON turns(conversation_id,id)")
            db.execute("CREATE TABLE IF NOT EXISTS feedback(turn_id INTEGER PRIMARY KEY, rating INTEGER NOT NULL, created REAL)")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def list(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT c.id,c.title,c.updated,COUNT(t.id) FROM conversations c "
                              "LEFT JOIN turns t ON t.conversation_id=c.id GROUP BY c.id "
                              "ORDER BY c.updated DESC LIMIT 100").fetchall()
        return [{"id": row[0], "title": row[1], "updated": row[2], "turns": row[3]} for row in rows]

    def get(self, conversation_id: str) -> dict:
        with self._connect() as db:
            conversation = db.execute("SELECT id,title,updated FROM conversations WHERE id=?",
                                      (conversation_id,)).fetchone()
            if not conversation:
                raise ValueError("会话不存在")
            rows = db.execute("SELECT id,question,final_answer,snapshot_id,evidence_json,created FROM turns "
                              "WHERE conversation_id=? ORDER BY id", (conversation_id,)).fetchall()
        return {"id": conversation[0], "title": conversation[1], "updated": conversation[2], "turns": [
            {"id": row[0], "question": row[1], "answer": row[2], "snapshot_id": row[3],
             "evidence": json.loads(row[4]), "created": row[5]} for row in rows]}

    def history(self, conversation_id: str | None) -> list[dict]:
        if not conversation_id:
            return []
        turns = self.get(conversation_id)["turns"][-MAX_HISTORY_TURNS:]
        selected, used = [], 0
        for turn in reversed(turns):
            size = len(turn["question"]) + len(turn["answer"])
            if selected and used + size > MAX_HISTORY_CHARS:
                break
            selected.append({"question": turn["question"], "answer": turn["answer"]})
            used += size
        return list(reversed(selected))

    def append(self, conversation_id: str | None, question: str, final_answer: str,
               snapshot_id: str, evidence: list[dict]) -> tuple[str, int]:
        conversation_id = conversation_id or uuid.uuid4().hex
        now = time.time()
        compact_evidence = [{key: item.get(key) for key in
                             ("chunk_id", "doc_id", "source", "page_start", "page_end",
                              "slide_start", "slide_end", "heading_path")} for item in evidence]
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO conversations VALUES(?,?,?)",
                       (conversation_id, question[:80], now))
            db.execute("UPDATE conversations SET updated=? WHERE id=?", (now, conversation_id))
            cursor = db.execute("INSERT INTO turns(conversation_id,question,final_answer,snapshot_id,evidence_json,created) "
                                "VALUES(?,?,?,?,?,?)", (conversation_id, question, final_answer, snapshot_id,
                                                       json.dumps(compact_evidence, ensure_ascii=False), now))
        return conversation_id, int(cursor.lastrowid)

    def delete(self, conversation_id: str) -> None:
        with self._connect() as db:
            if not db.execute("SELECT 1 FROM conversations WHERE id=?", (conversation_id,)).fetchone():
                raise ValueError("会话不存在")
            db.execute("DELETE FROM feedback WHERE turn_id IN (SELECT id FROM turns WHERE conversation_id=?)",
                       (conversation_id,))
            db.execute("DELETE FROM turns WHERE conversation_id=?", (conversation_id,))
            db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))

    def rate(self, turn_id: int, rating: int) -> None:
        if rating not in {-1, 1}:
            raise ValueError("反馈只能为好评或差评")
        with self._connect() as db:
            if not db.execute("SELECT 1 FROM turns WHERE id=?", (turn_id,)).fetchone():
                raise ValueError("问答记录不存在")
            db.execute("INSERT INTO feedback(turn_id,rating,created) VALUES(?,?,?) "
                       "ON CONFLICT(turn_id) DO UPDATE SET rating=excluded.rating,created=excluded.created",
                       (turn_id, rating, time.time()))

    def feedback_counts(self) -> dict[str, int]:
        with self._connect() as db:
            rows = db.execute("SELECT rating,COUNT(*) FROM feedback GROUP BY rating").fetchall()
        counts = dict(rows)
        return {"positive": counts.get(1, 0), "negative": counts.get(-1, 0)}
