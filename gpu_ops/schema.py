"""Shared single-host GPU tables in the gpu_ops PostgreSQL database."""

from sqlalchemy import Boolean, Column, DateTime, Integer, MetaData, String, Table, Text, Uuid, func

metadata = MetaData(schema="scheduler")
profiles = Table(
    "gpu_profiles", metadata, Column("name", String(64), primary_key=True),
    Column("memory_mib", Integer, nullable=False), Column("margin_mib", Integer, nullable=False),
    Column("exclusive", Boolean, nullable=False),
)
requests = Table(
    "gpu_requests", metadata, Column("id", Uuid, primary_key=True),
    Column("project", String(64), nullable=False), Column("profile", String(64), nullable=False),
    Column("state", Text, nullable=False), Column("reason", Text),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("touched_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)
leases = Table(
    "gpu_leases", metadata, Column("device", String(96), primary_key=True),
    Column("request_id", Uuid), Column("token", Uuid), Column("holder_pid", Integer),
    Column("holder_start", Text), Column("child_pid", Integer), Column("child_start", Text),
    Column("project", String(64)), Column("expires_at", DateTime(timezone=True)),
)
events = Table(
    "gpu_events", metadata, Column("device", String(96), primary_key=True),
    Column("reason", Text), Column("level", Text, nullable=False),
    Column("changed_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)
