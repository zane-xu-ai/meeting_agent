"""One fenced lease per GPU, with live capacity checks and cross-project fairness."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from sqlalchemy import case, create_engine, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config import settings
from gpu_ops import schema as s


LEASE_TTL = timedelta(seconds=45)
WAITING_TTL = timedelta(minutes=2)


def now():
    return datetime.now(timezone.utc)


def start_id(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def alive(pid, marker):
    return bool(pid and marker and start_id(pid) == marker)


@dataclass(frozen=True)
class Memory:
    device: str
    total_mib: int
    free_mib: int
    compute_pids: tuple[int, ...]


def nvml_memory(device=None):
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(device) if device else pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        identifier = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(identifier, bytes):
            identifier = identifier.decode()
        pids = tuple(sorted(item.pid for item in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)))
        return Memory(identifier, memory.total // 2**20, memory.free // 2**20, pids)
    finally:
        pynvml.nvmlShutdown()


class Scheduler:
    def __init__(self, dsn, device=None, sensor=nvml_memory):
        self.engine = create_engine(dsn, pool_pre_ping=True, hide_parameters=True)
        self.device = device
        self.sensor = sensor

    def initialize(self):
        with self.engine.begin() as connection:
            connection.execute(text("CREATE SCHEMA IF NOT EXISTS scheduler"))
            s.metadata.create_all(connection)

    def configure(self, name, memory, margin, exclusive=False):
        if not name or len(name) > 64 or not 0 < memory <= 131072 or not 512 <= margin <= 16384:
            raise ValueError("invalid profile or memory budget")
        values = {"memory_mib": memory, "margin_mib": margin, "exclusive": exclusive}
        with self.engine.begin() as connection:
            connection.execute(pg_insert(s.profiles).values(name=name, **values).on_conflict_do_update(
                index_elements=["name"], set_=values,
            ))

    def sample(self):
        samples = []
        for index in range(3):
            if index:
                time.sleep(.2)
            samples.append(self.sensor(self.device))
        latest = samples[-1]
        if len({item.device for item in samples}) != 1:
            raise RuntimeError("GPU changed during sampling")
        return Memory(latest.device, latest.total_mib, min(item.free_mib for item in samples),
                      tuple(sorted({pid for item in samples for pid in item.compute_pids})))

    @staticmethod
    def _can_run(profile, memory):
        return (memory.free_mib >= profile["memory_mib"] + profile["margin_mib"]
                and (not profile["exclusive"] or not memory.compute_pids))

    def acquire(self, request_id, project, profile_name):
        if not project or len(project) > 64:
            raise ValueError("invalid project name")
        measurement = self.sample()
        device = measurement.device
        time_now = now()
        with self.engine.begin() as connection:
            connection.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:device, 0))"), {"device": device})
            profiles = {row["name"]: row for row in connection.execute(select(s.profiles)).mappings()}
            profile = profiles.get(profile_name)
            if not profile:
                raise ValueError("GPU 模型预算尚未配置")
            request = pg_insert(s.requests).values(id=request_id, project=project, profile=profile_name, state="waiting")
            connection.execute(request.on_conflict_do_update(
                index_elements=["id"], set_={"touched_at": time_now, "state": case(
                    (s.requests.c.state == "released", "waiting"), else_=s.requests.c.state,
                )},
            ))
            own = connection.execute(select(s.requests).where(s.requests.c.id == request_id)).mappings().one()
            if own["project"] != project or own["profile"] != profile_name:
                raise ValueError("GPU 请求身份不相符")
            if own["state"] in {"cancelled", "released"}:
                return None, own["state"]

            lease = connection.execute(select(s.leases).where(s.leases.c.device == device)).mappings().one_or_none()
            if lease and lease["token"]:
                if lease["expires_at"] > time_now:
                    return self._waiting(connection, request_id, "managed_busy")
                if alive(lease["holder_pid"], lease["holder_start"]) or alive(lease["child_pid"], lease["child_start"]):
                    return self._waiting(connection, request_id, "stale_process")
                if lease["child_pid"] in measurement.compute_pids:
                    return self._waiting(connection, request_id, "stale_process")
                connection.execute(update(s.requests).where(s.requests.c.id == lease["request_id"]).values(
                    state="waiting", reason="lease_expired",
                ))
                connection.execute(update(s.leases).where(s.leases.c.device == device).values(
                    request_id=None, token=None, expires_at=None, holder_pid=None, holder_start=None,
                    child_pid=None, child_start=None,
                ))

            pending = connection.execute(select(s.requests).where(
                s.requests.c.state == "waiting", s.requests.c.touched_at >= time_now - WAITING_TTL,
            ).order_by(s.requests.c.created_at, s.requests.c.id)).mappings().all()
            candidates = [row for row in pending if row["profile"] in profiles and self._can_run(profiles[row["profile"]], measurement)]
            if not self._can_run(profile, measurement):
                return self._waiting(connection, request_id, "external_memory" if measurement.compute_pids else "capacity")

            oldest = candidates[0] if candidates else None
            if oldest and time_now - oldest["created_at"] < timedelta(minutes=5) and lease:
                oldest = next((row for row in candidates if row["project"] != lease["project"]), oldest)
            if oldest and oldest["id"] != request_id:
                return self._waiting(connection, request_id, "fair_queue")
            latest = self.sensor(device)
            if not self._can_run(profile, latest):
                return self._waiting(connection, request_id, "capacity_changed")
            token = uuid.uuid4()
            values = dict(request_id=request_id, token=token, holder_pid=os.getpid(), holder_start=start_id(os.getpid()),
                          child_pid=None, child_start=None, project=project, expires_at=time_now + LEASE_TTL)
            connection.execute(pg_insert(s.leases).values(device=device, **values).on_conflict_do_update(
                index_elements=["device"], set_=values,
            ))
            connection.execute(update(s.requests).where(s.requests.c.id == request_id).values(state="running", reason=None))
            return (device, request_id, token), None

    @staticmethod
    def _waiting(connection, request_id, reason):
        connection.execute(update(s.requests).where(s.requests.c.id == request_id, s.requests.c.state == "waiting")
                           .values(reason=reason))
        return None, reason

    def child(self, lease, pid):
        with self.engine.begin() as connection:
            row = connection.execute(update(s.leases).where(s.leases.c.device == lease[0], s.leases.c.token == lease[2])
                                     .values(child_pid=pid, child_start=start_id(pid))).rowcount
            if not row:
                raise RuntimeError("GPU lease lost")

    def heartbeat(self, lease):
        with self.engine.begin() as connection:
            return bool(connection.execute(update(s.leases).where(
                s.leases.c.device == lease[0], s.leases.c.token == lease[2], s.leases.c.expires_at > func.now(),
            ).values(expires_at=now() + LEASE_TTL)).rowcount)

    def release(self, lease, *, retry=False):
        with self.engine.begin() as connection:
            row = connection.execute(update(s.leases).where(s.leases.c.device == lease[0], s.leases.c.token == lease[2])
                                     .values(request_id=None, token=None, holder_pid=None, holder_start=None,
                                             child_pid=None, child_start=None, expires_at=None)).rowcount
            if row:
                connection.execute(update(s.requests).where(s.requests.c.id == lease[1]).values(
                    state="waiting" if retry else "released", reason=None, touched_at=func.now(),
                ))
            return bool(row)

    def cancel(self, request_id):
        with self.engine.begin() as connection:
            connection.execute(update(s.requests).where(s.requests.c.id == request_id, s.requests.c.state == "waiting")
                               .values(state="cancelled", reason=None))

    def snapshot(self):
        hardware = self.sensor(self.device)
        with self.engine.connect() as connection:
            waiting = list(connection.execute(select(s.requests).where(
                s.requests.c.state == "waiting", s.requests.c.touched_at >= now() - WAITING_TTL,
            ).order_by(s.requests.c.created_at)).mappings())
            lease = connection.execute(select(s.leases).where(s.leases.c.device == hardware.device)).mappings().one_or_none()
            alert = connection.execute(select(s.events).where(s.events.c.device == hardware.device)).mappings().one_or_none()
        return {
            "device": hardware.device, "free_mib": hardware.free_mib, "total_mib": hardware.total_mib,
            "compute_processes": len(hardware.compute_pids), "waiting": [
                {"project": row["project"], "profile": row["profile"], "reason": row["reason"],
                 "wait_seconds": max(0, int((now() - row["created_at"]).total_seconds()))}
                for row in waiting
            ],
            "lease": {"project": lease["project"], "expires_at": lease["expires_at"]} if lease and lease["token"] else None,
            "alert": {"level": alert["level"], "reason": alert["reason"]} if alert else None,
        }

    def observe(self):
        try:
            snapshot = self.snapshot()
        except Exception:
            return self._update_alert(self.device or "gpu0", "warning", "monitor_unavailable")
        lease = snapshot["lease"]
        external_wait = max((row["wait_seconds"] for row in snapshot["waiting"]
                             if row["reason"] in {"external_memory", "capacity_changed"}), default=0)
        if snapshot["alert"] and snapshot["alert"]["reason"] == "controlled_oom":
            with self.engine.connect() as connection:
                changed = connection.scalar(select(s.events.c.changed_at).where(s.events.c.device == snapshot["device"]))
            if changed and now() - changed < timedelta(minutes=5):
                return None
        if lease and lease["expires_at"] < now():
            level, reason = "critical", "lease_expired"
        elif external_wait >= 900:
            level, reason = "critical", "external_wait"
        elif external_wait >= 300:
            level, reason = "warning", "external_wait"
        else:
            level, reason = "ok", None
        return self._update_alert(snapshot["device"], level, reason)

    def report_oom(self, device):
        return self._update_alert(device, "critical", "controlled_oom")

    def _update_alert(self, device, level, reason):
        with self.engine.begin() as connection:
            previous = connection.execute(select(s.events).where(s.events.c.device == device).with_for_update()).mappings().one_or_none()
            if previous and (previous["level"], previous["reason"]) == (level, reason):
                return None
            values = {"level": level, "reason": reason, "changed_at": func.now()}
            connection.execute(pg_insert(s.events).values(device=device, **values).on_conflict_do_update(
                index_elements=["device"], set_=values,
            ))
            return level, reason


@lru_cache(maxsize=4)
def _configured(url, device):
    return Scheduler(url, device)


def configured():
    return _configured(settings.gpu_database_url, settings.gpu_device_uuid or None) if settings.gpu_database_url else None
