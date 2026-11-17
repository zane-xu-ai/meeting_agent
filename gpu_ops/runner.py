"""Run a child process inside a renewable GPU lease."""

from __future__ import annotations

import os
import signal
import subprocess
import time


def _stop(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=6)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_leased(scheduler, lease, command, *, timeout=900, stdout=None, stderr=None, extra_env=None):
    if not scheduler.heartbeat(lease):
        raise RuntimeError("GPU lease lost before startup")
    environment = {**os.environ, **(extra_env or {}), "CUDA_VISIBLE_DEVICES": lease[0]}
    with subprocess.Popen(command, start_new_session=True, env=environment, stdout=stdout, stderr=stderr) as child:
        try:
            scheduler.child(lease, child.pid)
            deadline = time.monotonic() + timeout
            renewal = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    return child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= renewal:
                        if not scheduler.heartbeat(lease):
                            raise RuntimeError("GPU lease lost while running")
                        renewal = time.monotonic() + 10
            raise TimeoutError("GPU command timed out")
        finally:
            _stop(child)
