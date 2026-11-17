"""Initialize, inspect and use the shared host GPU coordinator."""

import argparse
import json
import sys
import time
import uuid

from gpu_ops.runner import run_leased
from gpu_ops.scheduler import configured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("init")
    sub.add_parser("status")
    sub.add_parser("watch")
    profile = sub.add_parser("profile")
    profile.add_argument("name")
    profile.add_argument("memory", type=int)
    profile.add_argument("--margin", type=int, default=1024)
    profile.add_argument("--exclusive", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--project", required=True)
    run.add_argument("--profile", required=True)
    run.add_argument("--wait", type=int, default=900)
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    scheduler = configured()
    if not scheduler:
        parser.error("GPU_DATABASE_URL is required")
    match args.mode:
        case "init":
            scheduler.initialize()
        case "profile":
            scheduler.configure(args.name, args.memory, args.margin, args.exclusive)
        case "status":
            print(json.dumps(scheduler.snapshot(), default=str, ensure_ascii=False, indent=2))
        case "watch":
            while True:
                try:
                    alert = scheduler.observe()
                    if alert:
                        print("GPU", *alert, flush=True)
                except Exception as error:
                    print("GPU monitor failure", type(error).__name__, file=sys.stderr, flush=True)
                time.sleep(5)
        case "run":
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            if not command:
                parser.error("command is required")
            request = uuid.uuid4()
            deadline = time.monotonic() + args.wait
            previous = None
            try:
                while True:
                    lease, reason = scheduler.acquire(request, args.project, args.profile)
                    if lease:
                        try:
                            raise SystemExit(run_leased(scheduler, lease, command))
                        finally:
                            scheduler.release(lease)
                    if reason != previous:
                        print("GPU waiting", reason, file=sys.stderr, flush=True)
                        previous = reason
                    if time.monotonic() >= deadline:
                        raise SystemExit("GPU wait timed out")
                    time.sleep(5)
            finally:
                scheduler.cancel(request)


if __name__ == "__main__":
    main()
