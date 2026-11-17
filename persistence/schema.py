"""PostgreSQL tables for media, task history and stage provenance."""

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, ForeignKey, Index, Integer,
    MetaData, String, Table, Text, UniqueConstraint, Uuid, func,
)
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData(schema="meeting")


def table(name, *columns):
    return Table(name, metadata, *columns)


def ident(name="id"):
    return Column(name, Uuid, primary_key=True)


def ref(name, parent, *, nullable=False, ondelete="RESTRICT"):
    return Column(name, Uuid, ForeignKey(f"meeting.{parent}.id", ondelete=ondelete), nullable=nullable)


def job_ref():
    return Column("job_id", String(32), ForeignKey("meeting.analysis_jobs.id", ondelete="CASCADE"), nullable=False)


def timestamp(name, *, default=False):
    return Column(name, DateTime(timezone=True), server_default=func.now() if default else None, nullable=not default)


workspaces = table("workspaces", ident(), Column("name", Text, nullable=False))
workspace_members = table(
    "workspace_members", ident(), ref("workspace_id", "workspaces"),
    Column("principal", Text, nullable=False), Column("role", Text, nullable=False),
    UniqueConstraint("workspace_id", "principal"),
)
artifacts = table(
    "artifacts", ident(), ref("workspace_id", "workspaces"),
    Column("sha256", String(64), nullable=False), Column("storage_key", Text, nullable=False),
    Column("size_bytes", BigInteger, nullable=False), Column("mime_type", Text),
    Column("kind", Text, nullable=False), timestamp("created_at", default=True),
    UniqueConstraint("workspace_id", "sha256"),
)
media_sources = table(
    "media_sources", ident(), ref("workspace_id", "workspaces"),
    Column("identity", String(64), nullable=False), Column("source_kind", Text, nullable=False),
    Column("original_filename", Text), Column("original_url", Text), Column("canonical_url", Text),
    Column("platform", Text), Column("platform_item_id", Text), Column("title", Text),
    Column("description", Text), Column("uploader", Text), timestamp("published_at"),
    timestamp("source_created_at"), timestamp("source_modified_at"),
    Column("timestamp_provenance", JSONB, nullable=False), timestamp("created_at", default=True),
    timestamp("updated_at", default=True), UniqueConstraint("workspace_id", "identity"),
)
media_versions = table(
    "media_versions", ident(), ref("source_id", "media_sources"),
    ref("artifact_id", "artifacts", nullable=True),
    Column("content_sha256", String(64), nullable=False), Column("size_bytes", BigInteger),
    Column("mime_type", Text), Column("container", Text), Column("audio_codec", Text),
    Column("video_codec", Text), Column("sample_rate_hz", Integer), Column("channels", Integer),
    Column("width", Integer), Column("height", Integer), Column("duration_ms", BigInteger),
    Column("etag", Text), timestamp("http_last_modified"), timestamp("fetched_at", default=True),
    Column("source_metadata_snapshot", JSONB), Column("probe_json", JSONB),
    UniqueConstraint("source_id", "content_sha256"),
)
submission_batches = table(
    "submission_batches", ident(), ref("workspace_id", "workspaces"),
    Column("idempotency_key", Text), Column("created_by", Text, nullable=False),
    timestamp("created_at", default=True), UniqueConstraint("workspace_id", "idempotency_key"),
)
analysis_jobs = table(
    "analysis_jobs", Column("id", String(32), primary_key=True),
    ref("workspace_id", "workspaces"), ref("source_id", "media_sources"),
    ref("media_version_id", "media_versions", nullable=True),
    ref("batch_id", "submission_batches", nullable=True),
    Column("status", Text, nullable=False), Column("category", Text),
    Column("input_location", Text), Column("request_config", JSONB, nullable=False),
    Column("cache_policy_snapshot", JSONB),
    Column("state", JSONB, nullable=False), timestamp("created_at", default=True),
    timestamp("started_at"), timestamp("finished_at"),
)
Index("ix_jobs_workspace_created", analysis_jobs.c.workspace_id, analysis_jobs.c.created_at.desc(), analysis_jobs.c.id.desc())
Index("ix_jobs_workspace_status", analysis_jobs.c.workspace_id, analysis_jobs.c.status)
stage_runs = table(
    "stage_runs", ident(), job_ref(),
    Column("stage", Text, nullable=False), Column("unit", Text, nullable=False),
    Column("attempt", Integer, nullable=False), Column("status", Text, nullable=False),
    Column("requested_model", Text), Column("actual_model", Text), Column("provider", Text),
    Column("input_fingerprint", Text), Column("provider_job_id", Text),
    Column("details", JSONB), timestamp("started_at", default=True), timestamp("finished_at"),
    UniqueConstraint("job_id", "stage", "unit", "attempt"),
)
stage_outputs = table(
    "stage_outputs", ident(), ref("stage_run_id", "stage_runs", ondelete="CASCADE"),
    ref("artifact_id", "artifacts", nullable=True), Column("kind", Text, nullable=False),
    Column("value", JSONB), Column("content_sha256", String(64)),
    timestamp("created_at", default=True), UniqueConstraint("stage_run_id", "kind"),
)
transcripts = table(
    "transcripts", ident(), job_ref(), ref("parent_id", "transcripts", nullable=True),
    Column("variant", Text, nullable=False), Column("provider", Text),
    Column("model", Text), Column("language", Text), Column("text", Text, nullable=False),
    Column("content_sha256", String(64), nullable=False), Column("duration_ms", BigInteger),
    timestamp("created_at", default=True), UniqueConstraint("job_id", "variant"),
)
transcript_segments = table(
    "transcript_segments", ident(), ref("transcript_id", "transcripts", ondelete="CASCADE"),
    Column("ordinal", Integer, nullable=False), Column("start_ms", BigInteger), Column("end_ms", BigInteger),
    Column("speaker", Text), Column("text", Text, nullable=False), Column("words_json", JSONB),
    UniqueConstraint("transcript_id", "ordinal"),
)
classifications = table(
    "classifications", ident(), job_ref(), ref("transcript_id", "transcripts", nullable=True),
    Column("category", Text, nullable=False), Column("confidence", JSONB),
    Column("reason", Text), Column("model", Text), Column("origin", Text, nullable=False),
    timestamp("created_at", default=True),
)
analyses = table(
    "analyses", ident(), job_ref(), ref("transcript_id", "transcripts", nullable=True),
    ref("classification_id", "classifications", nullable=True),
    Column("revision", Integer, nullable=False), Column("category", Text, nullable=False),
    Column("title", Text), Column("summary_text", Text), Column("overall_score", Integer),
    Column("actual_model", Text), Column("result_json", JSONB), Column("report_md", Text, nullable=False),
    Column("questions_md", Text), timestamp("created_at", default=True),
    UniqueConstraint("job_id", "revision"),
)
analysis_questions = table(
    "analysis_questions", ident(), ref("analysis_id", "analyses", ondelete="CASCADE"),
    Column("ordinal", Integer, nullable=False), Column("question_text", Text, nullable=False),
    Column("topic", Text), Column("time_range", Text), Column("start_ms", BigInteger), Column("end_ms", BigInteger),
    Column("candidate_answer", Text), Column("recommended_answer", Text),
    Column("score", Integer), Column("improvement", JSONB), UniqueConstraint("analysis_id", "ordinal"),
)
job_events = table(
    "job_events", Column("seq", BigInteger, primary_key=True, autoincrement=True),
    job_ref(), Column("event_type", Text, nullable=False), Column("payload", JSONB, nullable=False),
    timestamp("created_at", default=True),
)
Index("ix_job_events_job_seq", job_events.c.job_id, job_events.c.seq)
audit_events = table(
    "audit_events", ident(), ref("workspace_id", "workspaces"), Column("event", Text, nullable=False),
    Column("payload", JSONB, nullable=False), timestamp("created_at", default=True),
)
