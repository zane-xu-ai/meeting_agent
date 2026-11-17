"""Memoize successful pipeline stages and preserve independent job history."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
import uuid
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, func, insert, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.pool import NullPool

from config import settings
from models.schemas import Transcript
from persistence.cache_schema import cache_entries as entries, cache_policies as policies
from persistence.repository import PERSONAL, digest_text, repository
from persistence.schema import analysis_jobs, audit_events, stage_outputs, stage_runs


DAYS = 86400
LAYERS = {
    "source": ("来源与下载", 6 * 3600), "preprocess": ("音频预处理", 7 * DAYS),
    "asr": ("语音转写", 90 * DAYS), "clean": ("清洗与切分", 30 * DAYS),
    "classify": ("分类", 90 * DAYS), "interview_batch": ("分段提问与评分", 30 * DAYS),
    "analysis": ("最终分析与报告", 90 * DAYS),
}
FILES = {"audio_path", "preprocessed_path", "asr_input", "source_asset", "result_path"}
TASK_CONTEXT: ContextVar[dict | None] = ContextVar("cache_task", default=None)


def fingerprint(data):
    return digest_text(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


@lru_cache(maxsize=1)
def code_version():
    root = Path(__file__).resolve().parents[1]
    files = ["agent/graph.py", "agent/pipeline.py", "agent/tools.py", "agent/prompts.py",
             "agent/classifier.py", "agent/llm_client.py", "agent/text_processor.py", "agent/asr_client.py",
             "utils/audio.py", "utils/media_network.py"]
    return fingerprint([hashlib.sha256((root / file).read_bytes()).hexdigest() for file in files])


@lru_cache(maxsize=1)
def ffmpeg_version():
    try:
        return subprocess.check_output(["ffmpeg", "-version"], timeout=4, text=True).splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def path_hash(value):
    if not value or not Path(value).is_file():
        raise FileNotFoundError("阶段输入文件不存在")
    with Path(value).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def policy_snapshot(connection):
    snapshot = {key: {"read": True, "write": True, "ttl": item[1]} for key, item in LAYERS.items()}
    for item in connection.execute(select(policies).where(policies.c.workspace_id == PERSONAL)).mappings():
        snapshot[item["stage"]] = {"read": item["read_enabled"], "write": item["write_enabled"],
                                   "ttl": item["ttl_seconds"]}
    return snapshot


def set_policy(repo, stage, read, write, ttl):
    with repo.engine.begin() as connection:
        repo._workspace(connection)
        values = {"read_enabled": read, "write_enabled": write, "ttl_seconds": ttl}
        connection.execute(pg_insert(policies).values(id=uuid.uuid4(), workspace_id=PERSONAL, stage=stage, **values)
                           .on_conflict_do_update(index_elements=["workspace_id", "stage"],
                                                  set_={**values, "updated_at": func.now()}))
        connection.execute(insert(audit_events).values(
            id=uuid.uuid4(), workspace_id=PERSONAL, event="cache_policy", payload={"stage": stage, **values},
        ))


def list_cache(repo, stage=None, limit=100):
    columns = [entries.c[name] for name in ("id", "stage", "key", "created_at", "expires_at", "invalidated_at", "hit_count", "last_hit_at")]
    query = select(*columns, func.pg_column_size(entries.c.payload).label("bytes")).where(entries.c.workspace_id == PERSONAL)
    if stage:
        query = query.where(entries.c.stage == stage)
    with repo.engine.connect() as connection:
        return {"policies": policy_snapshot(connection),
                "labels": {key: label for key, (label, _) in LAYERS.items()},
                "entries": list(map(dict, connection.execute(query.order_by(entries.c.created_at.desc()).limit(limit)).mappings()))}


def cache_preview(repo, identifier):
    with repo.engine.connect() as connection:
        payload = connection.scalar(select(entries.c.payload).where(entries.c.workspace_id == PERSONAL, entries.c.id == identifier))
    if payload is None:
        return None
    # Paths are derived from private hashes. Show a bounded preview for text/metadata results.
    return json.dumps(payload, ensure_ascii=False, indent=2)[:36000]


def cache_file(repo, identifier):
    with repo.engine.connect() as connection:
        payload = connection.scalar(select(entries.c.payload).where(
            entries.c.workspace_id == PERSONAL, entries.c.id == identifier,
        ))
    if payload:
        for key in ("source_asset", "audio_path", "preprocessed_path"):
            if key in payload:
                path = (repo.media_root / payload[key]["archive"]).resolve()
                if path.is_relative_to(repo.media_root) and path.is_file():
                    return path


def invalidate(repo, stage=None, identifier=None):
    query = update(entries).where(entries.c.workspace_id == PERSONAL, entries.c.invalidated_at.is_(None))
    if stage:
        query = query.where(entries.c.stage == stage)
    if identifier:
        query = query.where(entries.c.id == identifier)
    with repo.engine.begin() as connection:
        count = connection.execute(query.values(invalidated_at=func.now())).rowcount
        repo._workspace(connection)
        connection.execute(insert(audit_events).values(
            id=uuid.uuid4(), workspace_id=PERSONAL, event="cache_invalidate",
            payload={"stage": stage, "id": str(identifier) if identifier else None, "count": count},
        ))
    return count


def cache_policy(repo, job_id, stage):
    with repo.engine.connect() as connection:
        snapshot = connection.scalar(select(analysis_jobs.c.cache_policy_snapshot).where(analysis_jobs.c.id == job_id))
        selected = snapshot or policy_snapshot(connection)
        policy = selected[stage]
        if stage == "analysis" and not all(selected[key]["read"] for key in ("classify", "interview_batch")):
            return {**policy, "read": False}
        return policy


def cache_enabled():
    return TASK_CONTEXT.get() is not None


def _get(repo, layer, key):
    with repo.engine.begin() as connection:
        return connection.scalar(update(entries).where(
            entries.c.workspace_id == PERSONAL, entries.c.stage == layer, entries.c.key == key,
            entries.c.invalidated_at.is_(None), entries.c.expires_at > func.now(),
        ).values(hit_count=entries.c.hit_count + 1, last_hit_at=func.now()).returning(entries.c.payload))


def _put(repo, layer, key, value, ttl):
    with repo.engine.begin() as connection:
        repo._workspace(connection)
        data = {"payload": value, "expires_at": datetime.now(timezone.utc) + timedelta(seconds=ttl),
                "hit_count": 0, "last_hit_at": None, "invalidated_at": None}
        connection.execute(pg_insert(entries).values(id=uuid.uuid4(), workspace_id=PERSONAL, stage=layer, key=key, **data)
                           .on_conflict_do_update(index_elements=["workspace_id", "stage", "key"],
                                                  set_={**data, "created_at": func.now()}))


@lru_cache(maxsize=4)
def lock_engine(url):
    # Held across await without occupying the pool used for stage commits.
    return create_engine(url, poolclass=NullPool, hide_parameters=True, isolation_level="AUTOCOMMIT")


def _record(repo, job, layer, key, status, payload=None):
    with repo.engine.begin() as connection:
        if connection.scalar(select(analysis_jobs.c.id).where(analysis_jobs.c.id == job)) is None:
            return
        stage_run_id = uuid.uuid4()
        connection.execute(insert(stage_runs).values(
            id=stage_run_id, job_id=job, stage=layer, unit=stage_run_id.hex, attempt=1,
            status=status, input_fingerprint=key, finished_at=func.now(),
        ))
        if payload is not None:
            connection.execute(insert(stage_outputs).values(
                id=uuid.uuid4(), stage_run_id=stage_run_id, kind=layer, value=payload,
                content_sha256=fingerprint(payload),
            ))


async def memoize(repo, job, layer, material, compute, *, encode=lambda value: value, decode=lambda value: value):
    key = fingerprint([layer, code_version(), material])
    policy = await asyncio.to_thread(cache_policy, repo, job, layer)

    async def load():
        payload = await asyncio.to_thread(_get, repo, layer, key) if policy["read"] else None
        if payload is not None:
            try:
                value = await asyncio.to_thread(decode, payload)
            except FileNotFoundError:
                return None
            await asyncio.to_thread(_record, repo, job, layer, key, "cached", payload)
            return value

    async def build():
        try:
            value = await compute()
            payload = await asyncio.to_thread(encode, value)
        except Exception:
            await asyncio.to_thread(_record, repo, job, layer, key, "failed")
            raise
        if policy["write"]:
            await asyncio.to_thread(_put, repo, layer, key, payload, policy["ttl"])
        await asyncio.to_thread(_record, repo, job, layer, key, "completed", payload)
        return value

    existing = await load()
    if existing is not None:
        return existing
    if not (policy["read"] and policy["write"]):
        return await build()

    connection = await asyncio.to_thread(lock_engine(repo.engine.url).connect)
    try:
        def try_lock():
            return connection.scalar(text("SELECT pg_try_advisory_lock(hashtextextended(:key, 0))"), {"key": key})

        while not await asyncio.to_thread(try_lock):
            await asyncio.sleep(.3)
        existing = await load()
        return existing if existing is not None else await build()
    finally:
        await asyncio.to_thread(connection.close)


def _encode(repo, value):
    payload = {}
    for key, item in value.items():
        if key in FILES and item and Path(item).is_file():
            _, relative, _ = repo.archive(Path(item))
            payload[key] = {"archive": relative, "suffix": Path(item).suffix}
            question_path = Path(item).with_name("question.md")
            if key == "result_path" and question_path.is_file():
                payload["questions_md"] = question_path.read_text(encoding="utf-8")
        elif key in FILES:
            continue
        elif isinstance(item, Transcript):
            payload[key] = item.model_dump(mode="json")
        else:
            payload[key] = item
    return payload


def _decode(repo, job, layer, payload):
    output = {}
    for key, item in payload.items():
        if key in FILES:
            source = repo.media_root / item["archive"]
            if not source.is_file():
                raise FileNotFoundError(source)
            root = settings.output_dir / "work" / job / layer
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            target = root / (key + item["suffix"])
            shutil.copyfile(source, target)
            output[key] = str(target)
            if key == "result_path" and payload.get("questions_md") is not None:
                target.with_name("question.md").write_text(payload["questions_md"], encoding="utf-8")
        elif key in {"transcript", "cleaned_transcript"} and item is not None:
            output[key] = Transcript.model_validate(item)
        elif key != "questions_md":
            output[key] = item
    return output


async def graph_stage(layer, function, state):
    repo = repository()
    if not repo:
        return await function(state)
    task_type = state["source_type"]
    if layer == "source" and task_type == "local_audio":
        return await function(state)
    if layer == "source":
        locator = await asyncio.to_thread(path_hash, state["source"]) if task_type.startswith("local") else state["source"]
        material = [locator, task_type, settings.max_upload_bytes, ffmpeg_version()]
    elif layer == "preprocess":
        material = [await asyncio.to_thread(path_hash, state.get("audio_path") or state.get("source_asset") or state["source"]),
                    task_type, settings.asr_sample_rate, ffmpeg_version()]
    elif layer == "asr":
        from agent.local_asr import LOCAL_MODEL, model_fingerprint
        material = [await asyncio.to_thread(path_hash, state["asr_input"]),
                    state.get("asr_model") or settings.asr_models, settings.asr_sample_rate,
                    model_fingerprint() if state.get("asr_model") == LOCAL_MODEL else None]
    elif layer == "clean":
        material = [state["transcript"].model_dump(mode="json"), settings.min_sentence_length]
    else:
        material = [state.get("transcript_text"), state.get("llm_model") or settings.llm_models,
                    settings.llm_base_url, settings.max_tokens_per_chunk, task_type]
    task = state["task_id"]
    token = TASK_CONTEXT.set(state)
    try:
        return await memoize(
            repo, task, layer, material, lambda: function(state),
            encode=lambda value: _encode(repo, value), decode=lambda data: _decode(repo, task, layer, data),
        )
    finally:
        TASK_CONTEXT.reset(token)


async def memoized_json(llm, prompt, validate):
    state = TASK_CONTEXT.get()
    repo = repository() if state else None
    if not repo:
        return await llm.chat_json(prompt)

    async def ask():
        data = await llm.chat_json(prompt)
        validate(data)
        return {"data": data, "model": llm.model}

    result = await memoize(repo, state["task_id"], "interview_batch",
                           [prompt, llm.base_url, llm.models], ask)
    llm.model = result["model"]
    return result["data"]


async def memoized_classification(transcript, model, call):
    state = TASK_CONTEXT.get()
    repo = repository() if state else None
    if not repo:
        return await call()
    return await memoize(repo, state["task_id"], "classify",
                         [transcript, model or settings.llm_models, settings.llm_base_url], call)
