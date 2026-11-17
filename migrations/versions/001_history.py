"""Initial media and history schema."""

from alembic import op

from persistence import schema


revision = "001_history"
down_revision = None

TABLES = (
    schema.workspaces, schema.workspace_members, schema.artifacts, schema.media_sources,
    schema.media_versions, schema.submission_batches, schema.analysis_jobs, schema.stage_runs,
    schema.stage_outputs, schema.transcripts, schema.transcript_segments, schema.classifications,
    schema.analyses, schema.analysis_questions, schema.job_events, schema.audit_events,
)


def upgrade():
    for table in TABLES:
        table.create(op.get_bind())


def downgrade():
    for table in reversed(TABLES):
        table.drop(op.get_bind())
