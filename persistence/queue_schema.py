"""Durable dispatch, leases and serialized pipeline checkpoints."""

from sqlalchemy import Column, Index, Integer, PrimaryKeyConstraint, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB

from persistence.schema import ident, job_ref, ref, table, timestamp


outbox = table(
    "outbox", ident(), job_ref(), Column("step", Integer, nullable=False),
    Column("routing_key", Text, nullable=False), Column("attempt", Integer, nullable=False),
    Column("status", Text, nullable=False), timestamp("available_at", default=True),
    timestamp("publish_after", default=True), timestamp("published_at"),
    Column("lease_token", Uuid), timestamp("lease_until"), timestamp("finished_at"),
    Column("error_code", Text),
)
Index("ix_outbox_dispatch", outbox.c.status, outbox.c.publish_after)
checkpoints = table(
    "job_checkpoints", job_ref(), Column("step", Integer, nullable=False),
    Column("payload", JSONB, nullable=False), timestamp("created_at", default=True),
    PrimaryKeyConstraint("job_id", "step"),
)

batch_receipts = table(
    "batch_receipts", ident(), ref("batch_id", "submission_batches"),
    Column("body", JSONB), Column("fingerprint", Text, nullable=False),
    timestamp("created_at", default=True),
)
