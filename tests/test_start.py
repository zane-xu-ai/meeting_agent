"""Exercise the Ubuntu launcher with isolated service and command doubles."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT = Path(__file__).resolve().parents[1]
COMMAND_DOUBLE = """
import os
from pathlib import Path
import sys

tool = Path(sys.argv[0]).name
args = sys.argv[1:]
events = Path(os.environ["TEST_EVENTS"])
with events.open("a") as log:
    print(tool, *args, file=log)

scenario = os.environ.get("TEST_SCENARIO", "")
if tool == "ss" and scenario == "occupied":
    print("LISTEN 0 128 127.0.0.1:19090")
if tool == "uv" and scenario == "sync-failed":
    sys.exit(1)
if tool == "systemctl":
    action = args[1]
    commands = events.read_text().splitlines()
    started = "systemctl --user start meeting-agent.service" in commands
    if action == "is-active":
        sys.exit(0 if scenario == "active" or (started and scenario != "failed") else 1)
    if action == "is-failed":
        sys.exit(0 if scenario == "failed" else 1)
if tool == "curl":
    if scenario == "retry" and events.read_text().count("curl ") == 1:
        sys.exit(7)
    print('{"status":"ok","api_key_configured":false}')
"""


class StartScriptTests(unittest.TestCase):
    def setUp(self):
        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.home = Path(temp_dir.name)
        self.project = self.home / "projects" / "meeting_agent"
        (self.project / "deploy").mkdir(parents=True)
        shutil.copy2(PROJECT / "start.sh", self.project)
        shutil.copy2(PROJECT / "deploy" / "meeting-agent.service", self.project / "deploy")

        tools = self.home / "bin"
        tools.mkdir()
        for tool in ("curl", "ffmpeg", "ffprobe", "flock", "journalctl", "ss", "systemctl", "uv"):
            executable = tools / tool
            executable.write_text(f"#!{sys.executable}\n{COMMAND_DOUBLE}")
            executable.chmod(0o700)
        self.events = self.home / "events"
        self.env = {
            **os.environ,
            "HOME": str(self.home),
            "PATH": f"{tools}:/usr/bin:/bin",
            "TEST_EVENTS": str(self.events),
            "XDG_CONFIG_HOME": str(self.home / ".config"),
            "XDG_RUNTIME_DIR": str(self.home),
        }

    def run_script(self, *args, scenario=""):
        return subprocess.run(
            ["bash", str(self.project / "start.sh"), *args],
            env={**self.env, "TEST_SCENARIO": scenario},
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )

    def test_start_syncs_and_retries_health_check_before_enabling_service(self):
        result = self.run_script(scenario="retry")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Meeting Agent 已就绪", result.stdout)
        events = self.events.read_text()
        self.assertEqual(events.count("curl "), 2)
        self.assertLess(events.index("uv sync --locked"), events.index("--user start "))
        self.assertLess(events.rindex("curl "), events.index("--user enable "))

    def test_active_service_is_not_restarted_or_resynced(self):
        result = self.run_script("start", scenario="active")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("健康检查", result.stdout)
        events = self.events.read_text()
        self.assertNotIn("uv sync", events)
        self.assertNotIn("--user start ", events)

    def test_occupied_port_stops_before_syncing(self):
        result = self.run_script("start", scenario="occupied")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TCP 19090 已被占用", result.stderr)
        self.assertNotIn("uv sync", self.events.read_text())

    def test_sync_failure_leaves_service_stopped(self):
        result = self.run_script("start", scenario="sync-failed")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("--user start ", self.events.read_text())

    def test_service_failure_does_not_enable_service(self):
        result = self.run_script("start", scenario="failed")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("./start.sh logs", result.stderr)
        events = self.events.read_text()
        self.assertIn("--user stop ", events)
        self.assertNotIn("--user enable ", events)


if __name__ == "__main__":
    unittest.main()
