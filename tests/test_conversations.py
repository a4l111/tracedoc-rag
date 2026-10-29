"""Shared conversation persistence and bounded history checks."""

import tempfile
import unittest
from pathlib import Path

from src.service.conversations import Conversations, MAX_HISTORY_CHARS, MAX_HISTORY_TURNS


class ConversationTests(unittest.TestCase):
    def test_append_restore_bound_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = Conversations(Path(folder) / "conversations.sqlite3")
            conversation_id = None
            for number in range(MAX_HISTORY_TURNS + 3):
                conversation_id, _ = store.append(conversation_id, f"问题 {number}", "答" * 10,
                                                  "snapshot", [{"chunk_id": "c", "text": "must not persist"}])
            restored = store.get(conversation_id)
            self.assertEqual(len(restored["turns"]), MAX_HISTORY_TURNS + 3)
            self.assertNotIn("text", restored["turns"][0]["evidence"][0])
            history = store.history(conversation_id)
            self.assertLessEqual(len(history), MAX_HISTORY_TURNS)
            self.assertLessEqual(sum(len(t["question"]) + len(t["answer"]) for t in history), MAX_HISTORY_CHARS)
            store.delete(conversation_id)
            self.assertEqual(store.list(), [])


if __name__ == "__main__":
    unittest.main()
