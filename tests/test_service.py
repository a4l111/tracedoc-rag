"""Small HTTP contract checks without external model calls."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.service import web


class ServiceTests(unittest.TestCase):
    def test_upload_parse_and_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with (patch.object(web, "RAW", root / "raw"), patch.object(web, "PARSED", root / "parsed"),
                  patch.object(web, "JOBS", root / "jobs"), patch.object(web, "SERVICE", root / "service"),
                  patch.object(web, "EVENTS", root / "service" / "events.jsonl"),
                  patch.object(web.executor, "submit") as submit):
                with TestClient(web.app) as client:
                    response = client.post("/api/upload", files={"file": ("note.txt", io.BytesIO("测试内容".encode()), "text/plain")})
                    self.assertEqual(response.status_code, 200)
                    job_id = response.json()["job_id"]
                    paths = submit.call_args.args[3]
                    web._parse_job(job_id, paths)
                    jobs = client.get("/api/jobs").json()
                    self.assertEqual(jobs[0]["state"], "completed")
                    document = client.get("/api/documents").json()[0]
                    source = client.get(f"/api/source/{document['doc_id']}")
                    self.assertEqual(source.status_code, 200)
                    self.assertIn("测试内容", source.text)
                    self.assertEqual(client.post("/api/build", json={"chunk_mode": "other"}).status_code, 400)
                    self.assertEqual(client.post("/api/build", json={"chunk_mode": "fixed"},
                                                 headers={"Origin": "http://other.example"}).status_code, 403)

    def test_recover_interrupted_job(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs = Path(folder) / "jobs"
            jobs.mkdir()
            (jobs / "abc.json").write_text(json.dumps({"job_id": "abc", "state": "running"}), encoding="utf-8")
            with patch.object(web, "JOBS", jobs):
                web._recover_jobs()
                self.assertEqual(json.loads((jobs / "abc.json").read_text(encoding="utf-8"))["state"], "interrupted")


if __name__ == "__main__":
    unittest.main()
