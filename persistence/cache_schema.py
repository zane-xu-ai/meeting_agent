"""Workspace cache policy and reusable, content-addressed stage outputs."""

from sqlalchemy import BigInteger, Boolean, Column, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

from persistence.schema import ident, ref, table, timestamp


cache_policies = table(
    "cache_policies", ident(), ref("workspace_id", "workspaces"),
    Column("stage", Text, nullable=False), Column("read_enabled", Boolean, nullable=False),
    Column("write_enabled", Boolean, nullable=False), Column("ttl_seconds", Integer, nullable=False),
    timestamp("updated_at", default=True), UniqueConstraint("workspace_id", "stage"),
)
cache_entries = table(
    "cache_entries", ident(), ref("workspace_id", "workspaces"),
    Column("stage", Text, nullable=False), Column("key", String(64), nullable=False),
    Column("payload", JSONB, nullable=False), Column("hit_count", BigInteger, nullable=False),
    timestamp("expires_at"), timestamp("invalidated_at"), timestamp("created_at", default=True),
    timestamp("last_hit_at"), UniqueConstraint("workspace_id", "stage", "key"),
)
