"""Durable media records and private archive references."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import create_engine, func, insert, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config import settings
from persistence import schema as s


PERSONAL = uuid.UUID("52cb033e-5c49-4413-b177-05f10619b8ab")
PRIVATE_STATE = {"source", "input_location", "result_path", "asr_result_path"}


def digest_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def public_state(state):
    return {key: value for key, value in state.items() if key not in PRIVATE_STATE}


def display_url(url):
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def probe(path):
    try:
        output = subprocess.run(
            ["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
             "-show_format", "-show_streams", "-of", "json", str(path)],
            capture_output=True, check=True, timeout=30,
        )
        data = json.loads(output.stdout)
        audio = next((entry for entry in data["streams"] if entry.get("codec_type") == "audio"), {})
        video = next((entry for entry in data["streams"] if entry.get("codec_type") == "video"), {})
        return {
            "container": data["format"].get("format_name"),
            "audio_codec": audio.get("codec_name"),
            "video_codec": video.get("codec_name"),
            "duration_ms": int(float(data["format"].get("duration") or 0) * 1000),
            "sample_rate_hz": int(audio["sample_rate"]) if audio.get("sample_rate") else None,
            "channels": audio.get("channels"),
            "width": video.get("width"), "height": video.get("height"),
            "probe_json": {key: values for key, values in (
                ("format", {key: value for key, value in data["format"].items() if key != "filename"}),
                ("streams", data["streams"]),
            )},
        }
    except (OSError, subprocess.SubprocessError, KeyError, ValueError):
        return {}


class Repository:
    def __init__(self, dsn: str, media_root: Path):
        self.engine = create_engine(dsn, pool_pre_ping=True, hide_parameters=True)
        self.media_root = media_root.resolve()

    def archive(self, path: Path, sha256: str | None = None):
        if not sha256:
            with path.open("rb") as source:
                sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        key = f"{PERSONAL.hex}/{sha256[:2]}/{sha256}"
        target = self.media_root / key
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not target.exists():
            with NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                tmp = Path(temporary.name)
                try:
                    with path.open("rb") as source:
                        shutil.copyfileobj(source, temporary)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                    os.replace(tmp, target)
                finally:
                    tmp.unlink(missing_ok=True)
        return sha256, key, target.stat().st_size

    @staticmethod
    def _workspace(connection):
        connection.execute(pg_insert(s.workspaces).values(id=PERSONAL, name="个人工作区").on_conflict_do_nothing())
        connection.execute(pg_insert(s.workspace_members).values(
            id=uuid.uuid4(), workspace_id=PERSONAL, principal="meeting", role="owner",
        ).on_conflict_do_nothing())

    @staticmethod
    def _event(connection, job_id, kind, payload):
        connection.execute(insert(s.job_events).values(job_id=job_id, event_type=kind, payload=public_state(payload)))

    def create(self, state, location, request_config, *, modified_ms=None, batch_id=None):
        local = state["source_type"].startswith("local") and location is not None
        filename = state.get("filename") if local or not location else None
        url = display_url(location) if location and not local else None
        archived = self.archive(Path(location), state.get("content_sha256")) if local else None
        # Retain distinct display names and independent URLs even when bytes match.
        identity = digest_text(json.dumps([filename, archived[0]] if archived else [location or state["task_id"]]))
        modified = None
        if modified_ms:
            try:
                modified = datetime.fromtimestamp(modified_ms / 1000, timezone.utc)
            except (OverflowError, OSError, ValueError):
                pass
        source = {
            "id": uuid.uuid4(), "workspace_id": PERSONAL, "identity": identity,
            "source_kind": state["source_type"], "original_filename": filename,
            "original_url": url, "canonical_url": url, "title": filename or url,
            "source_modified_at": modified,
            "timestamp_provenance": {"source_modified_at": "browser"} if modified else {},
        }
        with self.engine.begin() as connection:
            self._workspace(connection)
            statement = pg_insert(s.media_sources).values(**source)
            source_id = connection.scalar(statement.on_conflict_do_update(
                index_elements=["workspace_id", "identity"], set_={"updated_at": func.now()},
            ).returning(s.media_sources.c.id))
            version = self._version(connection, source_id, archived, filename) if archived else None
            connection.execute(insert(s.analysis_jobs).values(
                id=state["task_id"], workspace_id=PERSONAL, source_id=source_id,
                media_version_id=version, status=state["status"], input_location=location,
                state=state, request_config=request_config, batch_id=batch_id,
            ))
            self._event(connection, state["task_id"], "status", state)

    def _version(self, connection, source_id, archive, filename=None, fields=None):
        checksum, key, size = archive
        mime = mimetypes.guess_type(filename or "")[0]
        artifact = pg_insert(s.artifacts).values(
            id=uuid.uuid4(), workspace_id=PERSONAL, sha256=checksum, storage_key=key,
            size_bytes=size, kind="source", mime_type=mime,
        )
        artifact_id = connection.scalar(artifact.on_conflict_do_update(
            index_elements=["workspace_id", "sha256"], set_={"storage_key": key},
        ).returning(s.artifacts.c.id))
        details = fields or {}
        query = pg_insert(s.media_versions).values(
            id=uuid.uuid4(), source_id=source_id, artifact_id=artifact_id,
            content_sha256=checksum, size_bytes=size, mime_type=mime, **details,
        )
        return connection.scalar(query.on_conflict_do_update(
            index_elements=["source_id", "content_sha256"], set_=details or {"artifact_id": artifact_id},
        ).returning(s.media_versions.c.id))

    def update(self, job_id, changes):
        with self.engine.begin() as connection:
            state = connection.scalar(select(s.analysis_jobs.c.state).where(s.analysis_jobs.c.id == job_id).with_for_update())
            if state is None:
                return
            state.update(changes)
            values = {"state": state, "status": state["status"], "category": state.get("rec_type")}
            if changes.get("status") in {"completed", "failed", "cancelled"}:
                values["finished_at"] = func.now()
            if changes.get("status") == "processing":
                values["started_at"] = func.coalesce(s.analysis_jobs.c.started_at, func.now())
            connection.execute(update(s.analysis_jobs).where(s.analysis_jobs.c.id == job_id).values(**values))
            self._event(connection, job_id, "status", changes)

    def record_result(self, job_id, result, state):
        """Publish the transcript and the report together before exposing success."""
        source = result.get("source_asset") or (result.get("source") if state["source_type"].startswith("local") else None)
        source = Path(source) if source and Path(source).is_file() else None
        media_details = probe(source) if source else {}
        source_meta = result.get("source_metadata") or {}
        media_details["source_metadata_snapshot"] = source_meta
        archived = self.archive(source, state.get("content_sha256")) if source else None

        report_path = Path(state["result_path"]) if state.get("result_path") else None
        report = result.get("output_text") or (report_path.read_text() if report_path and report_path.is_file() else "")
        if not report.strip():
            raise ValueError("分析报告为空")
        question_path = report_path.with_name("question.md") if report_path and state.get("rec_type") == "interview" else None
        questions = question_path.read_text() if question_path and question_path.is_file() else None

        data = result.get("analysis_data", {})
        scoring = data.get("a_data", data)
        with self.engine.begin() as connection:
            job = connection.execute(select(s.analysis_jobs).where(s.analysis_jobs.c.id == job_id).with_for_update()).mappings().one()
            if job["status"] == "completed":
                return
            if archived:
                version = self._version(connection, job["source_id"], archived, state.get("filename"), media_details)
                connection.execute(update(s.analysis_jobs).where(s.analysis_jobs.c.id == job_id).values(media_version_id=version))
            title = source_meta.get("title") or state.get("video_title") or state.get("filename")
            if source_meta:
                try:
                    published_at = datetime.strptime(source_meta.get("upload_date", ""), "%Y%m%d").replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    published_at = None
                connection.execute(update(s.media_sources).where(s.media_sources.c.id == job["source_id"]).values(
                    title=title, description=source_meta.get("description"), uploader=source_meta.get("uploader"),
                    published_at=published_at, updated_at=func.now(),
                ))
            parent = None
            if result.get("transcript") and not result.get("clean_asr_cache_hit"):
                parent = self._transcript(connection, job_id, "raw", result["transcript"].formatted_text, result["transcript"], result)
            transcript_id = self._transcript(
                connection, job_id, "cleaned", result.get("transcript_text", ""),
                result.get("cleaned_transcript"), result, parent,
            )
            category = result.get("rec_type") or state.get("rec_type") or "general"
            classification_id = uuid.uuid4()
            classification = result.get("classification_data") or {}
            connection.execute(insert(s.classifications).values(
                id=classification_id, job_id=job_id, transcript_id=transcript_id, category=category, origin="imported" if state["source_type"] == "legacy" else "automatic",
                model=classification.get("model"), reason=classification.get("reason"), confidence=classification.get("confidence"),
            ))
            summary = scoring.get("overall_summary") or scoring.get("summary")
            analysis_id = uuid.uuid4()
            connection.execute(insert(s.analyses).values(
                id=analysis_id, job_id=job_id, revision=1, category=category, transcript_id=transcript_id,
                classification_id=classification_id, title=title, report_md=report, questions_md=questions,
                result_json=data or None, summary_text=summary if isinstance(summary, str) else None,
                overall_score=scoring.get("overall_score"), actual_model=result.get("llm_model_used"),
            ))
            self._questions(connection, analysis_id, data)
            completed = {**state, "status": "completed", "progress": 100, "step": "分析完成",
                         "rec_type": category, "llm_model_used": result.get("llm_model_used") or state.get("llm_model_used")}
            connection.execute(update(s.analysis_jobs).where(s.analysis_jobs.c.id == job_id).values(
                state=completed, category=category, status="completed", finished_at=func.now(),
            ))
            self._event(connection, job_id, "status", completed)

    @staticmethod
    def _transcript(connection, job, variant, text, transcript, result, parent=None):
        if not text:
            return None
        identifier = uuid.uuid4()
        connection.execute(insert(s.transcripts).values(
            id=identifier, job_id=job, variant=variant, parent_id=parent, text=text,
            content_sha256=digest_text(text), duration_ms=transcript.duration_ms if transcript else None,
            model=result.get("asr_model_used"),
        ))
        if transcript and transcript.sentences:
            connection.execute(insert(s.transcript_segments), [
                {"id": uuid.uuid4(), "transcript_id": identifier, "ordinal": index,
                 "start_ms": sentence.begin_time, "end_ms": sentence.end_time,
                 "speaker": sentence.speaker, "text": sentence.text}
                for index, sentence in enumerate(transcript.sentences, 1)
            ])
        return identifier

    @staticmethod
    def _questions(connection, analysis_id, data):
        scores = {str(row.get("id")): row for row in data.get("a_data", {}).get("question_analyses", [])}
        rows = []
        for index, question in enumerate(data.get("q_data", {}).get("questions", []), 1):
            grade = scores.get(str(question.get("id")), {})
            rows.append({
                "id": uuid.uuid4(), "analysis_id": analysis_id, "ordinal": index,
                "question_text": question["question"], "topic": question.get("context"),
                "time_range": question.get("time_range"), "score": grade.get("score"),
                "candidate_answer": grade.get("candidate_answer"), "recommended_answer": grade.get("recommended_answer"),
                "improvement": grade.get("improvement"),
            })
        if rows:
            connection.execute(insert(s.analysis_questions), rows)

    def media_file(self, job_id):
        statement = (
            select(s.artifacts.c.storage_key, s.artifacts.c.mime_type)
            .select_from(s.analysis_jobs).join(s.media_versions, s.analysis_jobs.c.media_version_id == s.media_versions.c.id)
            .join(s.artifacts, s.media_versions.c.artifact_id == s.artifacts.c.id)
        ).where(
            s.analysis_jobs.c.id == job_id, s.analysis_jobs.c.workspace_id == PERSONAL,
        )
        with self.engine.connect() as connection:
            row = connection.execute(statement).one_or_none()
        if row:
            path = (self.media_root / row.storage_key).resolve()
            if path.is_relative_to(self.media_root) and path.is_file():
                return path, row.mime_type

    def task(self, job_id):
        with self.engine.connect() as connection:
            return connection.scalar(select(s.analysis_jobs.c.state).where(
                s.analysis_jobs.c.id == job_id, s.analysis_jobs.c.workspace_id == PERSONAL,
            ))

    def content(self, job_id, kind):
        table = s.transcripts if kind == "asr" else s.analyses
        column = {"asr": s.transcripts.c.text, "result": s.analyses.c.report_md, "questions": s.analyses.c.questions_md}[kind]
        predicate = [table.c.job_id == job_id, s.analysis_jobs.c.workspace_id == PERSONAL]
        if kind == "asr":
            predicate.append(s.transcripts.c.variant == "cleaned")
        with self.engine.connect() as connection:
            return connection.scalar(select(column).join(s.analysis_jobs, table.c.job_id == s.analysis_jobs.c.id).where(
                *predicate,
            ).order_by(table.c.created_at.desc()).limit(1))

    def history(self, limit, cursor=None, *, status=None, category=None, search=None):
        jobs, sources = s.analysis_jobs, s.media_sources
        query = select(jobs.c.id, jobs.c.created_at, jobs.c.finished_at, jobs.c.state, sources.c.title).join(sources)
        query = query.where(jobs.c.workspace_id == PERSONAL)
        if cursor:
            query = query.where(tuple_(jobs.c.created_at, jobs.c.id) < tuple_(*cursor))
        if status:
            query = query.where(jobs.c.status == status)
        if category:
            query = query.where(jobs.c.category == category)
        if search:
            query = query.where(func.coalesce(sources.c.title, "").icontains(search, autoescape=True) |
                                func.coalesce(sources.c.original_filename, "").icontains(search, autoescape=True) |
                                func.coalesce(sources.c.canonical_url, "").icontains(search, autoescape=True))
        with self.engine.connect() as connection:
            rows = connection.execute(query.order_by(jobs.c.created_at.desc(), jobs.c.id.desc()).limit(limit + 1)).mappings().all()
        return [{**row, "state": public_state(row["state"])} for row in rows]

    def detail(self, job_id):
        jobs, sources, versions = s.analysis_jobs, s.media_sources, s.media_versions
        with self.engine.connect() as connection:
            job = connection.execute(select(jobs).where(jobs.c.id == job_id, jobs.c.workspace_id == PERSONAL)).mappings().one_or_none()
            if job is None:
                return None
            source = connection.execute(select(sources).where(sources.c.id == job["source_id"])).mappings().one()
            media = connection.execute(select(versions).where(versions.c.id == job["media_version_id"])).mappings().one_or_none()
            analysis = connection.execute(select(s.analyses).where(s.analyses.c.job_id == job_id).order_by(s.analyses.c.revision.desc()).limit(1)).mappings().one_or_none()
            if analysis:
                questions = connection.execute(select(s.analysis_questions).where(
                    s.analysis_questions.c.analysis_id == analysis["id"],
                ).order_by(s.analysis_questions.c.ordinal)).mappings().all()
            else:
                questions = []
            segments = connection.execute(select(s.transcript_segments).join(s.transcripts).where(
                s.transcripts.c.job_id == job_id, s.transcripts.c.variant == "cleaned",
            ).order_by(s.transcript_segments.c.ordinal)).mappings().all()
        return {"task": public_state(job["state"]), "source": dict(source), "media": dict(media) if media else None,
                "analysis": dict(analysis) if analysis else None, "questions": list(map(dict, questions)),
                "segments": list(map(dict, segments))}


@lru_cache(maxsize=4)
def _repository(dsn, root):
    return Repository(dsn, root)


def repository():
    return _repository(settings.database_url, settings.media_storage_dir) if settings.database_url else None
