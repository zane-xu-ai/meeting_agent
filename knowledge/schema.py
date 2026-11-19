"""Versioned events and evidence-backed facts, scoped to a workspace."""

from sqlalchemy import Column, Integer, Text, String, Boolean, Date, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import JSONB
from persistence.schema import ident, ref, table, timestamp

state = table('knowledge_state', ref('workspace_id', 'workspaces'), Column('revision', Integer, nullable=False),
              UniqueConstraint('workspace_id'))
sessions = table('meeting_sessions', ident(), ref('workspace_id', 'workspaces'),
    Column('identity_key', Text, nullable=False), Column('title', Text, nullable=False),
    Column('category', Text, nullable=False), Column('event_date', Date),
    timestamp('event_start_at'), timestamp('event_end_at'), Column('timezone', Text, nullable=False),
    Column('precision', Text, nullable=False), Column('time_status', Text, nullable=False),
    Column('revision', Integer, nullable=False), Column('deleted', Boolean, nullable=False),
    timestamp('updated_at', default=True), UniqueConstraint('workspace_id', 'identity_key'))
Index('ix_meeting_events_date', sessions.c.workspace_id, sessions.c.event_date)
recordings = table('meeting_recordings', ident(), ref('session_id', 'meeting_sessions', ondelete='CASCADE'),
    ref('analysis_id', 'analyses', ondelete='CASCADE'), ref('media_version_id', 'media_versions', nullable=True),
    Column('current', Boolean, nullable=False), UniqueConstraint('analysis_id'))
time_candidates = table('meeting_time_candidates', ident(), ref('session_id', 'meeting_sessions', ondelete='CASCADE'),
    Column('raw', Text), Column('source', Text, nullable=False), Column('value', Date), Column('status', Text, nullable=False))
people = table('people', ident(), ref('workspace_id', 'workspaces'), Column('name', Text, nullable=False),
    Column('organization', Text), Column('active', Boolean, nullable=False))
aliases = table('person_aliases', ident(), ref('person_id', 'people', ondelete='CASCADE'),
    Column('alias', Text, nullable=False), Column('confirmed', Boolean, nullable=False),
    UniqueConstraint('person_id', 'alias'))
evidence = table('evidence_spans', ident(), ref('workspace_id', 'workspaces'),
    ref('session_id', 'meeting_sessions', ondelete='CASCADE'), ref('analysis_id', 'analyses', nullable=True),
    ref('segment_id', 'transcript_segments', nullable=True), Column('kind', Text, nullable=False),
    Column('quote', Text, nullable=False), Column('start_ms', Integer), Column('end_ms', Integer),
    Column('source_revision', Integer, nullable=False), Column('text_hash', String(64), nullable=False))
speakers = table('speaker_links', ident(), ref('session_id', 'meeting_sessions', ondelete='CASCADE'),
    ref('transcript_id', 'transcripts'), Column('label', Text, nullable=False), ref('person_id', 'people', nullable=True),
    Column('confirmed', Boolean, nullable=False), UniqueConstraint('session_id', 'transcript_id', 'label'))
participants = table('meeting_participants', ident(), ref('session_id', 'meeting_sessions', ondelete='CASCADE'),
    ref('person_id', 'people', nullable=True), ref('speaker_id', 'speaker_links', nullable=True),
    Column('display_name', Text, nullable=False), Column('status', Text, nullable=False),
    ref('evidence_id', 'evidence_spans'), Column('active', Boolean, nullable=False))
topics = table('meeting_topics', ident(), ref('session_id', 'meeting_sessions', ondelete='CASCADE'),
    Column('title', Text, nullable=False), Column('summary', Text, nullable=False),
    ref('evidence_id', 'evidence_spans'), Column('active', Boolean, nullable=False))
builds = table('knowledge_builds', ident(), ref('analysis_id', 'analyses', ondelete='CASCADE'),
    ref('session_id', 'meeting_sessions', nullable=True), Column('status', Text, nullable=False),
    Column('extractor_version', Text, nullable=False), Column('error_code', Text),
    timestamp('updated_at', default=True), UniqueConstraint('analysis_id', 'extractor_version'))
queries = table('query_runs', ident(), ref('workspace_id', 'workspaces'), Column('conversation', Text, nullable=False),
    Column('question', Text, nullable=False), Column('plan', JSONB, nullable=False), Column('response', JSONB, nullable=False),
    Column('knowledge_revision', Integer, nullable=False), Column('duration_ms', Integer, nullable=False),
    timestamp('created_at', default=True))
Index('ix_queries_scope_conversation', queries.c.workspace_id, queries.c.conversation, queries.c.created_at)
entities = table('knowledge_entities', ident(), ref('workspace_id', 'workspaces'),
    Column('kind', Text, nullable=False), Column('name', Text, nullable=False), UniqueConstraint('workspace_id', 'kind', 'name'))
relations = table('knowledge_relations', ident(), ref('workspace_id', 'workspaces'),
    ref('source_id', 'knowledge_entities'), ref('target_id', 'knowledge_entities'),
    Column('kind', Text, nullable=False), ref('session_id', 'meeting_sessions', ondelete='CASCADE'),
    ref('evidence_id', 'evidence_spans'), timestamp('valid_from'), timestamp('valid_to'),
    Column('active', Boolean, nullable=False))
chunks = table('knowledge_chunks', ident(), ref('workspace_id', 'workspaces'),
    ref('session_id', 'meeting_sessions', ondelete='CASCADE'), ref('evidence_id', 'evidence_spans'),
    Column('text', Text, nullable=False), Column('embedding', JSONB), Column('embedding_model', Text),
    Column('source_revision', Integer, nullable=False))
feedback = table('query_feedback', ident(), ref('workspace_id', 'workspaces'), ref('query_id', 'query_runs'),
    Column('helpful', Boolean, nullable=False), Column('note', Text), timestamp('created_at', default=True))

TABLES = (state, sessions, recordings, time_candidates, people, aliases, evidence, speakers, participants,
          topics, builds, queries, entities, relations, chunks, feedback)
