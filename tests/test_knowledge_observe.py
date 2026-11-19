"""Observe real samples, persist failures and never claim a premature completion."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from knowledge.observe import observe


class ObservationTests(unittest.TestCase):
    def test_durable_report_waits_for_duration_and_counts_failures(self):
        now = [0]

        def pause(seconds):
            now[0] += seconds

        def probe(base):
            if now[0] == 1:
                raise ConnectionError("private information must not be logged")
            return {"health": {"status": "ok"}, "coverage": {"revision": int(now[0])}}

        with TemporaryDirectory() as root:
            path = Path(root) / "report.json"
            result = observe(
                path,
                duration=2,
                interval=1,
                probe=probe,
                clock=lambda: now[0],
                pause=pause,
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["elapsed_seconds"], 2)
            self.assertEqual(result["samples"], 3)
            self.assertEqual(result["errors"], 1)
            self.assertEqual(result["revision_changes"], 1)
            self.assertNotIn("private information", path.read_text())
            self.assertEqual(json.loads(path.read_text()), result)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
