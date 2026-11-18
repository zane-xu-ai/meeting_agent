"""Regression coverage for completed interview downloads, without model calls."""

import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import server


ROOT = Path(__file__).resolve().parents[1]


class Buttons(HTMLParser):
    def __init__(self):
        super().__init__()
        self.buttons = []

    def handle_starttag(self, tag, attrs):
        if tag == "button":
            self.buttons.append(dict(attrs))


class QuestionDownloadTests(unittest.TestCase):
    def test_main_page_exposes_question_download_only_for_interviews(self):
        source = (ROOT / "frontend/index.html").read_text()
        parser = Buttons()
        parser.feed(source)
        questions = [item for item in parser.buttons if item.get("@click") == "downloadQuestions"]
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["v-if"],
                         "activeResultTab === 'analysis' && meta.rec_type === 'interview'")
        self.assertIn("submitUrl, downloadResult, downloadQuestions, copyResult", source)
        self.assertIn("/api/questions/${encodeURIComponent(task.task_id)}/download", source)
        self.assertTrue(any(item.get("@click") == "downloadResult" for item in parser.buttons))

    def test_existing_interview_questions_download_uses_persisted_content(self):
        repo = Mock()
        repo.task.return_value = {
            "task_id": "existing-interview", "status": "completed", "rec_type": "interview",
            "filename": "面试录音.m4a",
        }
        repo.content.return_value = "# 面试问题清单\n\n### Q1. 已提取的问题"
        with (
            patch.object(server, "repository", return_value=repo),
            patch.object(server.settings, "workspace_token", ""),
            patch.object(server, "_process") as analyze,
            TestClient(server.app, base_url="http://localhost", client=("127.0.0.1", 1234)) as client,
        ):
            response = client.get("/api/questions/existing-interview/download")
        self.assertEqual(response.status_code, 200)
        self.assertIn("已提取的问题", response.text)
        self.assertIn("attachment;", response.headers["content-disposition"])
        self.assertIn("questions.md", response.headers["content-disposition"])
        repo.content.assert_called_once_with("existing-interview", "questions")
        analyze.assert_not_called()
