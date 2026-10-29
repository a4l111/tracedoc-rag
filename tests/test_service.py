"""Small HTTP contract checks without external model calls."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastapi import HTTPException

from src.service import web


class ServiceTests(unittest.TestCase):
    def test_model_admin_is_local_only(self):
        web._require_local_admin(SimpleNamespace(client=SimpleNamespace(host="127.0.0.1")))
        with self.assertRaises(HTTPException) as denied:
            web._require_local_admin(SimpleNamespace(client=SimpleNamespace(host="192.168.1.20")))
        self.assertEqual(denied.exception.status_code, 403)

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

    def test_jobs_are_ordered_by_update_time_not_random_id(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs = Path(folder)
            (jobs / "z.json").write_text(json.dumps({"job_id": "z", "updated_at": "2026-01-01T00:00:00Z"}), encoding="utf-8")
            (jobs / "a.json").write_text(json.dumps({"job_id": "a", "updated_at": "2026-01-02T00:00:00Z"}), encoding="utf-8")
            with patch.object(web, "JOBS", jobs):
                self.assertEqual([item["job_id"] for item in web.jobs()], ["a", "z"])

    def test_retry_build_and_read_parse_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = root / "jobs"
            jobs.mkdir()
            failed = {"job_id": "build1", "kind": "build", "state": "failed", "errors": 1,
                      "retry": {"chunk_mode": "structured"}}
            partial = {"job_id": "parse1", "kind": "parse", "state": "partial", "errors": 1,
                       "failed_files": ["bad.pdf"], "retry": {"paths": ["bad.pdf"], "describe_images": False}}
            (jobs / "build1.json").write_text(json.dumps(failed), encoding="utf-8")
            (jobs / "parse1.json").write_text(json.dumps(partial), encoding="utf-8")
            with patch.object(web, "JOBS", jobs), patch.object(web.executor, "submit") as submit:
                with TestClient(web.app) as client:
                    retried = client.post("/api/jobs/build1/retry")
                    self.assertEqual(retried.status_code, 200)
                    self.assertEqual(submit.call_args.args[2], web._build_job)
                    errors = client.get("/api/jobs/parse1/errors").json()
                    self.assertEqual(errors["errors"], [{"type": "parse_failed", "file": "bad.pdf"}])

    def test_retry_rejects_completed_or_legacy_job(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs = Path(folder) / "jobs"
            jobs.mkdir()
            (jobs / "completed.json").write_text(json.dumps({"job_id": "completed", "kind": "build",
                "state": "completed", "retry": {"chunk_mode": "fixed"}}), encoding="utf-8")
            (jobs / "legacy.json").write_text(json.dumps({"job_id": "legacy", "kind": "build",
                "state": "failed"}), encoding="utf-8")
            with patch.object(web, "JOBS", jobs):
                with TestClient(web.app) as client:
                    self.assertEqual(client.post("/api/jobs/completed/retry").status_code, 409)
                    self.assertEqual(client.post("/api/jobs/legacy/retry").status_code, 409)

    def test_local_model_settings_never_return_secret(self):
        visible = {"deepseek": {"provider": "deepseek", "base_url": "https://api.deepseek.com",
                                "model": "deepseek-chat", "timeout": 30, "key_status": "encrypted_store"}}
        with patch.object(web, "public_settings", return_value=visible), \
             patch.object(web, "save_role"), patch.object(web, "set_secret") as save_secret:
            with TestClient(web.app) as client:
                response = client.get("/api/admin/model-settings")
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("secret-value", response.text)
                saved = client.put("/api/admin/model-settings", json={
                    "role": "deepseek", "provider": "deepseek", "base_url": "https://api.deepseek.com",
                    "model": "deepseek-chat", "timeout": 30, "api_key": "secret-value",
                })
                self.assertEqual(saved.status_code, 200)
                save_secret.assert_called_once_with("deepseek", "secret-value")
                self.assertNotIn("secret-value", saved.text)

    def test_categories_images_feedback_and_usage_are_metadata_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parsed = root / "parsed" / "doc-1"
            (parsed / "images").mkdir(parents=True)
            (parsed / "images" / "chart.png").write_bytes(b"image")
            (parsed / "manifest.json").write_text(json.dumps({
                "doc_id": "doc-1", "source_file": "survey.pdf", "source_type": "pdf",
                "sha256": "source-hash", "image_assets": [
                    {"path": "images/chart.png", "page": 9, "kind": "embedded"}]}, ensure_ascii=False), encoding="utf-8")
            service = root / "service"
            with patch.object(web, "PARSED", root / "parsed"), patch.object(web, "SERVICE", service), \
                 patch.object(web, "CATEGORIES", service / "categories.json"), \
                 patch.object(web, "EVENTS", service / "events.jsonl"):
                with TestClient(web.app) as client:
                    self.assertEqual(client.put("/api/documents/doc-1/category", json={"category": "调查报告"}).status_code, 200)
                    self.assertEqual(client.get("/api/documents").json()[0]["category"], "调查报告")
                    self.assertEqual(client.get("/api/documents").json()[0]["images"], 1)
                    self.assertEqual(len(web._assets("doc-1", 9, 9, None, None, "source-hash")), 1)
                    self.assertEqual(web._assets("doc-1", 9, 9, None, None, "other-hash"), [])
                    conversations = web.Conversations(service / "conversations.sqlite3")
                    _, turn_id = conversations.append(None, "question", "answer", "snapshot", [])
                    self.assertEqual(client.post(f"/api/turns/{turn_id}/feedback", json={"rating": 1}).status_code, 200)
                    self.assertEqual(client.post(f"/api/turns/{turn_id}/feedback", json={"rating": -1}).status_code, 200)
                    self.assertEqual(client.post(f"/api/turns/{turn_id}/feedback", json={"rating": 0}).status_code, 400)
                    web._token_event({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
                    metrics = client.get("/api/metrics").json()
                    self.assertEqual(metrics["deepseek_tokens"], 15)
                    self.assertEqual(metrics["feedback"], {"positive": 0, "negative": 1})
                    self.assertNotIn("question", (service / "events.jsonl").read_text(encoding="utf-8"))

    def test_build_job_records_real_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            snapshot = root / "indexes" / "snapshot-1"
            snapshot.mkdir(parents=True)
            (snapshot / "build_report.json").write_text(json.dumps({
                "complete": True, "chunk_count": 2, "error_count": 0}), encoding="utf-8")
            with patch.object(web, "JOBS", root / "jobs"), patch.object(web, "INDEXES", root / "indexes"), \
                 patch.object(web, "SERVICE", root / "service"), patch.object(web, "EVENTS", root / "service" / "events.jsonl"):
                job_id = web._new_job("build")
                def fake_build(*args, **kwargs):
                    kwargs["on_progress"]("dense", 1, 2, 0)
                    record = web._load_job(job_id)
                    self.assertEqual((record["stage"], record["completed"], record["total"]), ("Qwen3 向量化", 1, 2))
                    return snapshot
                with patch.object(web, "build_snapshot", side_effect=fake_build):
                    web._build_job(job_id, "structured")
                self.assertEqual(web._load_job(job_id)["state"], "completed")


if __name__ == "__main__":
    unittest.main()
