"""Transactional stage transitions shared by HTTP, dispatcher and workers."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import and_, func, insert, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config import settings
from models.schemas import Transcript
from persistence.queue_schema import batch_receipts, checkpoints, outbox
from persistence.repository import PERSONAL
from persistence.schema import analysis_jobs as jobs
from persistence.schema import job_events, submission_batches


# Stage 0 fetches inputs, 1 normalizes them; CPU work can advance while cloud
# or GPU workers are waiting on a long model call.
STEPS = (
    ("extract_audio", "io"), ("preprocess", "cpu"), ("asr_transcribe", "cloud"),
    ("text_clean", "cpu"), ("agent_loop", "cloud"), ("format_output", "cpu"),
)
TERMINAL = {"completed", "failed", "cancelled"}
PUBLISH_RESERVATION = timedelta(seconds=20)
REPUBLISH_INTERVAL = timedelta(minutes=15)
LEASE = timedelta(seconds=90)
RETRIES = (10, 60, 300)


def utcnow():
    return datetime.now(timezone.utc)


def enabled():
    return bool(settings.amqp_url)


def stage_route(job, step):
    from agent.local_asr import LOCAL_MODEL
    return "gpu" if STEPS[step][0] == "asr_transcribe" and job["request_config"].get("asr_model") == LOCAL_MODEL else STEPS[step][1]


def enqueue(connection, job_id, step=0, *, attempt=1, delay=0, route=None):
    queue = route or STEPS[step][1]
    ready = utcnow() + timedelta(seconds=delay)
    dispatch_id = uuid.uuid4()
    connection.execute(insert(outbox).values(
        id=dispatch_id, job_id=job_id, step=step, routing_key=queue, attempt=attempt,
        status="pending", available_at=ready, publish_after=ready,
    ))
    return dispatch_id


def reserve_publications(repo, limit=30):
    with repo.engine.begin() as connection:
        rows = connection.execute(select(outbox).join(jobs, outbox.c.job_id == jobs.c.id).where(
            or_(jobs.c.status.not_in(TERMINAL), outbox.c.routing_key == "dead"),
            outbox.c.available_at <= func.now(), outbox.c.publish_after <= func.now(),
            or_(outbox.c.status.in_(("pending", "published")),
                and_(outbox.c.status == "running", outbox.c.lease_until <= func.now())),
        ).order_by(outbox.c.publish_after, outbox.c.id).limit(limit).with_for_update(of=outbox, skip_locked=True)).mappings().all()
        if rows:
            connection.execute(update(outbox).where(outbox.c.id.in_([row["id"] for row in rows]))
                               .values(publish_after=utcnow() + PUBLISH_RESERVATION))
        return list(map(dict, rows))


def published(repo, identifier):
    with repo.engine.begin() as connection:
        row = _locked(connection, identifier)
        if row and row["status"] in {"pending", "published"}:
            connection.execute(update(outbox).where(outbox.c.id == identifier).values(
                status="completed" if row["routing_key"] == "dead" else "published",
                published_at=func.now(), publish_after=utcnow() + REPUBLISH_INTERVAL,
            ))


def _locked(connection, identifier):
    return connection.execute(select(outbox).where(outbox.c.id == identifier).with_for_update()).mappings().one_or_none()


def _job(connection, identifier):
    return connection.execute(select(jobs).where(jobs.c.id == identifier).with_for_update()).mappings().one()


def _change(repo, connection, job, **values):
    state = {**job["state"], **values}
    update_values = {"state": state, "status": state["status"], "category": state.get("rec_type")}
    if values.get("status") in TERMINAL:
        update_values["finished_at"] = func.now()
    if values.get("status") == "processing":
        update_values["started_at"] = func.coalesce(jobs.c.started_at, func.now())
    connection.execute(update(jobs).where(jobs.c.id == job["id"]).values(**update_values))
    repo._event(connection, job["id"], "status", values)
    return state


def claim(repo, identifier, route=None):
    with repo.engine.begin() as connection:
        row = _locked(connection, identifier)
        if not row or row["status"] not in {"pending", "published", "running"} or row["available_at"] > utcnow():
            return None
        if route and row["routing_key"] != route:
            return None
        if row["status"] == "running" and row["lease_until"] > utcnow():
            return None
        job = _job(connection, row["job_id"])
        if job["status"] in TERMINAL or job["status"] == "cancel_requested":
            if job["status"] == "cancel_requested":
                _change(repo, connection, job, status="cancelled", step="已取消")
            connection.execute(update(outbox).where(outbox.c.id == identifier).values(status="cancelled"))
            return None
        token = uuid.uuid4()
        connection.execute(update(outbox).where(outbox.c.id == identifier).values(
            status="running", lease_token=token, lease_until=utcnow() + LEASE, publish_after=utcnow() + LEASE,
        ))
        saved = connection.scalar(select(checkpoints.c.payload).where(
            checkpoints.c.job_id == job["id"], checkpoints.c.step == row["step"] - 1,
        ))
        state = _change(repo, connection, job, status="processing", step=STEPS[row["step"]][0])
        return {**row, "lease_token": token, "job": {**job, "state": state}, "checkpoint": saved}


def heartbeat(repo, work):
    with repo.engine.begin() as connection:
        count = connection.execute(update(outbox).where(
            outbox.c.id == work["id"], outbox.c.lease_token == work["lease_token"], outbox.c.status == "running",
            outbox.c.lease_until > func.now(),
        ).values(lease_until=utcnow() + LEASE, publish_after=utcnow() + LEASE)).rowcount
        return bool(count)


def _owned(connection, work):
    row = _locked(connection, work["id"])
    return row and row["status"] == "running" and row["lease_token"] == work["lease_token"] and row["lease_until"] > utcnow()


def progress(repo, work, values):
    with repo.engine.begin() as connection:
        if not _owned(connection, work):
            return
        job = _job(connection, work["job_id"])
        if job["status"] == "processing":
            _change(repo, connection, job, **values)


def encode(state):
    return {key: item.model_dump(mode="json") if isinstance(item, Transcript) else item
            for key, item in state.items()}


def decode(state):
    return {key: Transcript.model_validate(item) if key in {"transcript", "cleaned_transcript"} and item is not None else item
            for key, item in state.items()}


def complete(repo, work, state):
    with repo.engine.begin() as connection:
        if not _owned(connection, work):
            return False
        job = _job(connection, work["job_id"])
        final = work["step"] == len(STEPS) - 1
        if job["status"] == "cancel_requested":
            _change(repo, connection, job, status="cancelled", step="已取消")
        elif final:
            repo.record_result(job["id"], state, job["state"], connection=connection)
        else:
            payload = encode(state)
            connection.execute(pg_insert(checkpoints).values(job_id=job["id"], step=work["step"], payload=payload)
                               .on_conflict_do_update(index_elements=["job_id", "step"], set_={"payload": payload}))
            next_step = work["step"] + 1
            enqueue(connection, job["id"], next_step, route=stage_route(job, next_step))
            _change(repo, connection, job, status="pending", step="等待下一阶段")
        connection.execute(update(outbox).where(outbox.c.id == work["id"])
                           .values(status="completed", finished_at=func.now(), lease_token=None))
        return True


GPU_WAIT_REASONS = {
    "managed_busy": "受管 GPU 正在运行", "stale_process": "等待过期任务释放 GPU",
    "external_memory": "外部进程占用显存", "capacity": "显存预算不足",
    "capacity_changed": "显存用量变化", "fair_queue": "其他项目优先",
    "monitor_unavailable": "调度器或显存监测暂不可用",
}


def defer_gpu(repo, work, reason):
    with repo.engine.begin() as connection:
        if not _owned(connection, work):
            return
        job = _job(connection, work["job_id"])
        if job["status"] == "cancel_requested":
            _change(repo, connection, job, status="cancelled", step="已取消")
            connection.execute(update(outbox).where(outbox.c.id == work["id"]).values(status="cancelled"))
            return
        ready = utcnow() + timedelta(seconds=10)
        connection.execute(update(outbox).where(outbox.c.id == work["id"]).values(
            status="pending", available_at=ready, publish_after=ready, lease_token=None, lease_until=None,
        ))
        _change(repo, connection, job, status="waiting_gpu", step=GPU_WAIT_REASONS.get(reason, reason),
                gpu_wait_reason=reason)


def retryable(error):
    visited = set()
    while error and id(error) not in visited:
        visited.add(id(error))
        if isinstance(error, (TimeoutError, httpx.TransportError)):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code == 429 or error.response.status_code >= 500
        error = error.__cause__ or error.__context__
    return False


def failed(repo, work, error):
    code = type(error).__name__
    with repo.engine.begin() as connection:
        if not _owned(connection, work):
            return
        job = _job(connection, work["job_id"])
        status = "dead"
        if job["status"] == "cancel_requested":
            _change(repo, connection, job, status="cancelled", step="已取消")
        elif work["attempt"] <= len(RETRIES) and retryable(error):
            delay = RETRIES[work["attempt"] - 1]
            enqueue(connection, job["id"], work["step"], attempt=work["attempt"] + 1, delay=delay, route=work["routing_key"])
            _change(repo, connection, job, status="retrying", step=f"{delay} 秒后重试", error_code=code)
            status = "retried"
        else:
            enqueue(connection, job["id"], work["step"], route="dead")
            _change(repo, connection, job, status="failed", step="处理失败",
                    error=getattr(error, "public_message", None) or "阶段失败，请查看任务记录", error_code=code)
        connection.execute(update(outbox).where(outbox.c.id == work["id"])
                           .values(status=status, finished_at=func.now(), error_code=code, lease_token=None))


def cancel(repo, job_id):
    with repo.engine.begin() as connection:
        job = connection.execute(select(jobs).where(jobs.c.id == job_id, jobs.c.workspace_id == PERSONAL)
                                 .with_for_update()).mappings().one_or_none()
        if job is None or job["status"] in TERMINAL:
            return False
        status = "cancel_requested" if job["status"] == "processing" else "cancelled"
        _change(repo, connection, job, status=status, step="等待当前阶段退出" if status == "cancel_requested" else "已取消")
        return True


def resume(repo, job_id):
    with repo.engine.begin() as connection:
        job = connection.execute(select(jobs).where(jobs.c.id == job_id, jobs.c.workspace_id == PERSONAL,
                                                    jobs.c.status == "failed").with_for_update()).mappings().one_or_none()
        if not job:
            return False
        step = connection.scalar(select(func.max(outbox.c.step)).where(outbox.c.job_id == job_id))
        if step is None:
            return False
        enqueue(connection, job_id, step, route=stage_route(job, step))
        _change(repo, connection, job, status="pending", step="等待续跑", error=None, error_code=None)
        return True


def events(repo, job_id, after):
    with repo.engine.connect() as connection:
        return list(map(dict, connection.execute(select(job_events).where(
            job_events.c.job_id == job_id, job_events.c.seq > after,
        ).order_by(job_events.c.seq).limit(100)).mappings()))


def create_batch(repo, fingerprint, key=None):
    with repo.engine.begin() as connection:
        repo._workspace(connection)
        identifier = uuid.uuid4()
        command = pg_insert(submission_batches).values(
            id=identifier, workspace_id=PERSONAL, created_by="meeting", idempotency_key=key,
        ).on_conflict_do_nothing().returning(submission_batches.c.id)
        created = connection.scalar(command)
        if created:
            connection.execute(insert(batch_receipts).values(id=identifier, batch_id=identifier, fingerprint=fingerprint))
            return identifier, None, True
        receipt = connection.execute(select(batch_receipts).join(submission_batches).where(
            submission_batches.c.workspace_id == PERSONAL, submission_batches.c.idempotency_key == key,
        )).mappings().one()
        if receipt["fingerprint"] != fingerprint:
            raise ValueError("此幂等键已用于不同的批次")
        return receipt["batch_id"], receipt["body"], False


def save_batch_receipt(repo, batch_id, body):
    with repo.engine.begin() as connection:
        connection.execute(update(batch_receipts).where(batch_receipts.c.id == batch_id).values(body=body))


def batch_status(repo, batch_id):
    with repo.engine.connect() as connection:
        record = connection.execute(select(batch_receipts).join(submission_batches).where(
            submission_batches.c.id == batch_id, submission_batches.c.workspace_id == PERSONAL,
        )).mappings().one_or_none()
        if not record:
            return None
        rows = connection.execute(select(jobs.c.state).where(
            jobs.c.batch_id == batch_id, jobs.c.workspace_id == PERSONAL,
        ).order_by(jobs.c.created_at, jobs.c.id)).scalars()
        from persistence.repository import public_state
        return {"batch_id": str(batch_id), "items": [public_state(state) for state in rows],
                "rejected": (record["body"] or {}).get("rejected", [])}
