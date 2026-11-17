"""History acceptance against a migrated disposable PostgreSQL database."""

import os
import unittest
import uuid
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

import server
from models.schemas import Sentence, Transcript
from persistence.repository import Repository


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"), "requires a disposable PostgreSQL database")
class HistoryTests(unittest.TestCase):
    def test_completed_record_survives_fresh_store_and_removed_exports(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            dsn = os.environ["TEST_DATABASE_URL"]
            repo = Repository(dsn, directory / "archive")
            task_id = uuid.uuid4().hex
            filename = f"{task_id}-example.wav"
            source = directory / filename
            with wave.open(str(source), "wb") as sample:
                sample.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                sample.writeframes(b"\0" * 1600)
            report = directory / "analyze.md"
            report.write_text("# 面试报告")
            questions = directory / "question.md"
            questions.write_text("### Q1. 示例问题")
            state = {"task_id": task_id, "status": "processing", "source_type": "local_audio",
                     "filename": filename, "rec_type": "interview", "result_path": str(report)}
            repo.create(state, str(source), {"asr_model": "test"}, modified_ms=1730000000000)
            transcript = Transcript(sentences=[Sentence(text="示例", begin_time=10, end_time=30)])
            result = {
                "source": str(source), "rec_type": "interview", "output_text": report.read_text(),
                "transcript": transcript, "cleaned_transcript": transcript, "transcript_text": transcript.formatted_text,
                "analysis_data": {"q_data": {"questions": [{"id": 1, "question": "示例问题"}]},
                                  "a_data": {"overall_score": 85, "question_analyses": [{"id": 1, "score": 85}]}},
            }
            repo.record_result(task_id, result, state)
            source.unlink()
            report.unlink()
            questions.unlink()
            reopened = Repository(dsn, directory / "archive")
            with (
                patch.object(server, "repository", return_value=reopened),
                patch.object(server.settings, "workspace_token", ""),
                TestClient(server.app, base_url="http://localhost", client=("127.0.0.1", 1234)) as client,
            ):
                item = client.get(f"/api/history/{task_id}").raise_for_status().json()
                self.assertEqual(item["questions"][0]["score"], 85)
                self.assertEqual(item["segments"][0]["start_ms"], 10)
                self.assertIsNone(item["source"]["source_created_at"])
                self.assertIsNotNone(item["source"]["source_modified_at"])
                self.assertEqual(item["media"]["sample_rate_hz"], 16000)
                self.assertEqual(client.get(f"/api/history/{task_id}/media").status_code, 200)
                self.assertIn("面试报告", client.get(f"/api/result/{task_id}/download").text)
                self.assertIn("示例问题", client.get(f"/api/questions/{task_id}/download").text)
                self.assertEqual(client.get("/api/history", params={"search": filename}).json()["items"][0]["id"], task_id)
