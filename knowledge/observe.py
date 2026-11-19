"""Content-free, restart-safe release observation; never fabricates user traffic."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


def sample(base_url):
    result = {}
    for key, path in [
        ("health", "/api/health"),
        ("coverage", "/api/knowledge/coverage"),
    ]:
        with urlopen(base_url + path, timeout=5) as response:
            result[key] = json.load(response)
    with urlopen(base_url + "/api/knowledge/metrics", timeout=5) as response:
        if "meeting_knowledge_revision" not in response.read().decode():
            raise ValueError("Invalid metrics response")
    result["metrics_ready"] = True
    return result


def observe(
    report,
    *,
    duration=86400,
    interval=60,
    base_url="http://127.0.0.1:19090",
    probe=sample,
    clock=time.monotonic,
    pause=time.sleep,
):
    if duration <= 0 or interval <= 0:
        raise ValueError("duration and interval must be positive")
    report = Path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    started = clock()
    data = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "samples": 0,
        "errors": 0,
        "revision_changes": 0,
        "elapsed_seconds": 0,
        "scope": "passive availability and coverage; mixed-traffic quality requires separate review",
    }
    previous = None
    while True:
        try:
            current = probe(base_url)
            rev = current["coverage"]["revision"]
            if previous is not None and rev != previous:
                data["revision_changes"] += 1
            previous = rev
            data["latest"] = current
            data.pop("last_error", None)
        except Exception as error:
            data["errors"] += 1
            data["last_error"] = type(error).__name__
        data["samples"] += 1
        data["elapsed_seconds"] = int(clock() - started)
        if clock() - started >= duration:
            data["status"] = "completed"
            data["finished_at"] = datetime.now(timezone.utc).isoformat()
        temporary = report.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        os.chmod(temporary, 0o600)
        temporary.replace(report)
        if data["status"] == "completed":
            return data
        pause(min(interval, duration - (clock() - started)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=86400)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--report", default="data/knowledge-observation.json")
    args = parser.parse_args()
    observe(args.report, duration=args.duration, interval=args.interval)
