"""Constrained natural-language-to-query AST. SQL is compiled, never supplied by users."""
from typing import Literal
from datetime import date,datetime,timezone
from pydantic import BaseModel,Field,ConfigDict,model_validator
from sqlalchemy import select,func,distinct
from knowledge import schema as k
from knowledge.planner import period
import asyncio

class AggregatePlan(BaseModel):
    model_config=ConfigDict(extra='forbid')
    metric:Literal['attendance_count']='attendance_count'
    group_by:Literal['person']='person'
    start:date
    stop:date
    category:Literal['meeting','interview','general','video']='meeting'
    top_n:int=Field(default=10,ge=1,le=20)
    @model_validator(mode='after')
    def valid_period(self):
        if self.start>=self.stop or (self.stop-self.start).days>3660:raise ValueError('日期区间无效')
        return self

# Keep joins and authorization predicates in one auditable compiler.
def execute(c,workspace,p):
    count=func.count(distinct(k.sessions.c.id)).label('meeting_count')
    statement=(select(k.people.c.id,k.people.c.name,count).select_from(k.participants)
        .join(k.people,k.people.c.id==k.participants.c.person_id)
        .join(k.sessions,k.sessions.c.id==k.participants.c.session_id)
        .where(k.people.c.workspace_id==workspace,k.sessions.c.workspace_id==workspace,
            k.people.c.active.is_(True),k.participants.c.active.is_(True),k.participants.c.status=='confirmed',
            k.sessions.c.deleted.is_(False),k.sessions.c.time_status=='confirmed',k.sessions.c.category==p.category,
            k.sessions.c.event_date>=p.start,k.sessions.c.event_date<p.stop)
        .group_by(k.people.c.id,k.people.c.name).order_by(count.desc(),k.people.c.id).limit(p.top_n))
    # A read-only transaction still has explicit timeout and row bounds.
    from sqlalchemy import text
    c.execute(text("SET LOCAL statement_timeout = '2000ms'"))
    return list(map(dict,c.execute(statement).mappings()))

def from_question(question,anchor=None,timezone_name='Asia/Shanghai',model=None):
    start,stop=period(question,anchor or datetime.now(timezone.utc),timezone_name)
    if model is None:
        from agent.llm_client import LLMClient
        model=LLMClient()
    prompt=('将问题转换为只读统计计划，只允许 metric=attendance_count、group_by=person、category=meeting/interview/general/video、top_n=1..20。'
            '只返回 JSON，禁止 SQL、工作区和任何额外字段。问题：'+question)
    payload=asyncio.run(model.chat_json(prompt))
    payload.update(start=start,stop=stop)
    return AggregatePlan.model_validate(payload)
