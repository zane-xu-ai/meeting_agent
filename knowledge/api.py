"""Workspace-scoped knowledge review and querying endpoints."""
import uuid
from datetime import date
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from persistence.repository import PERSONAL, repository
from knowledge import schema as k
from knowledge.store import confirm, owned, revision

router=APIRouter(prefix='/api/knowledge')

def repo_required():
    repo=repository()
    if repo is None: raise HTTPException(503,'知识查询需要 PostgreSQL')
    return repo

@router.get('/sessions')
def sessions(limit:int=50,offset:int=0):
    repo=repo_required()
    with repo.engine.connect() as c:
        rows=list(c.execute(select(k.sessions).where(k.sessions.c.workspace_id==PERSONAL,k.sessions.c.deleted.is_(False))
                 .order_by(k.sessions.c.updated_at.desc(),k.sessions.c.id).limit(max(1,min(limit,100))).offset(max(0,offset))).mappings())
        count=c.scalar(select(func.count()).select_from(k.sessions).where(k.sessions.c.workspace_id==PERSONAL,k.sessions.c.deleted.is_(False)))
    return {'items':list(map(dict,rows)),'total':count}

@router.get('/sessions/{session_id}')
def detail(session_id:uuid.UUID):
    repo=repo_required()
    with repo.engine.connect() as c:
        try: session=owned(c,PERSONAL,session_id)
        except LookupError as error: raise HTTPException(404,str(error)) from error
        facts={name:list(map(dict,c.execute(select(table).where(table.c.session_id==session_id)).mappings()))
               for name,table in [('participants',k.participants),('topics',k.topics),('evidence',k.evidence),('time_candidates',k.time_candidates),('recordings',k.recordings)]}
    return {'session':dict(session),**facts}

class Confirmation(BaseModel):
    event_date:date|None=None
    names:list[str]|None=Field(default=None,max_length=200)

@router.put('/sessions/{session_id}/confirmation')
def confirmation(session_id:uuid.UUID,body:Confirmation):
    if body.names is not None and any(not name.strip() or len(name)>100 for name in body.names):
        raise HTTPException(422,'姓名不能为空或超过 100 字')
    try: confirm(repo_required(),PERSONAL,session_id,event_date=body.event_date,names=[name.strip() for name in body.names] if body.names is not None else None)
    except LookupError as error: raise HTTPException(404,str(error)) from error
    except ValueError as error: raise HTTPException(409,str(error)) from error
    return {'updated':True}

@router.get('/coverage')
def coverage():
    repo=repo_required()
    with repo.engine.connect() as c:
        filters=(k.sessions.c.workspace_id==PERSONAL,k.sessions.c.deleted.is_(False))
        total=c.scalar(select(func.count()).select_from(k.sessions).where(*filters))
        dates=c.scalar(select(func.count()).select_from(k.sessions).where(*filters,k.sessions.c.time_status=='confirmed'))
        return {'sessions':total,'confirmed_dates':dates,'unknown_or_candidate_dates':total-dates,'revision':revision(c,PERSONAL)}

class QueryRequest(BaseModel):
    question:str=Field(min_length=1,max_length=2000)
    conversation:str=Field(default='default',min_length=1,max_length=100)
    timezone:str=Field(default='Asia/Shanghai',max_length=80)

@router.post('/query')
def ask(body:QueryRequest):
    from knowledge.query import query
    return query(repo_required(),PERSONAL,body.question,body.conversation,timezone_name=body.timezone)

@router.get('/evidence/{evidence_id}')
def get_evidence(evidence_id:uuid.UUID):
    with repo_required().engine.connect() as c:
        row=c.execute(select(k.evidence).join(k.sessions,k.sessions.c.id==k.evidence.c.session_id).where(
            k.evidence.c.workspace_id==PERSONAL,k.evidence.c.id==evidence_id,k.sessions.c.deleted.is_(False))).mappings().one_or_none()
        if row is None:raise HTTPException(404,'引用不存在或已撤销')
        return dict(row)

class FeedbackRequest(BaseModel):
    query_id:uuid.UUID
    helpful:bool
    note:str|None=Field(default=None,max_length=1000)

@router.post('/feedback')
def feedback(body:FeedbackRequest):
    from sqlalchemy import insert
    with repo_required().engine.begin() as c:
        if not c.scalar(select(k.queries.c.id).where(k.queries.c.workspace_id==PERSONAL,k.queries.c.id==body.query_id)):
            raise HTTPException(404,'查询不存在')
        c.execute(insert(k.feedback).values(id=uuid.uuid4(),workspace_id=PERSONAL,query_id=body.query_id,helpful=body.helpful,note=body.note))
    return {'saved':True}

@router.delete('/sessions/{session_id}')
def delete_session(session_id:uuid.UUID):
    from sqlalchemy import update
    from knowledge.store import bump
    with repo_required().engine.begin() as c:
        try:owned(c,PERSONAL,session_id,lock=True)
        except LookupError as error:raise HTTPException(404,str(error)) from error
        c.execute(update(k.sessions).where(k.sessions.c.id==session_id).values(deleted=True))
        bump(c,PERSONAL)
    return {'deleted':True}
