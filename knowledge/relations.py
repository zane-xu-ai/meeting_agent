"""Evidence-scoped temporal relations and reversible person aliases."""
from datetime import datetime,timezone
from typing import Literal
import uuid
from pydantic import BaseModel,Field,ConfigDict
from sqlalchemy import select,insert,update,or_
from sqlalchemy.dialects.postgresql import insert as upsert
from knowledge import schema as k
from knowledge.store import owned,bump

class RelationInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    session_id:uuid.UUID
    evidence_id:uuid.UUID
    source_name:str=Field(min_length=1,max_length=100)
    target_name:str=Field(min_length=1,max_length=100)
    source_kind:Literal['project','topic','decision','action','person','organization']
    target_kind:Literal['project','topic','decision','action','person','organization']
    kind:Literal['discusses','decides','supersedes','assigned_to','follows_up','belongs_to']
    valid_from:datetime|None=None
    valid_to:datetime|None=None

def add(repo,workspace,body):
    for value in (body.valid_from,body.valid_to):
        if value is not None and value.tzinfo is None:raise ValueError('有效时间需要包含时区')
    if body.valid_from and body.valid_to and body.valid_from>=body.valid_to:raise ValueError('有效时间区间无效')
    with repo.engine.begin() as c:
        owned(c,workspace,body.session_id,lock=True)
        if not c.scalar(select(k.evidence.c.id).where(k.evidence.c.workspace_id==workspace,k.evidence.c.session_id==body.session_id,k.evidence.c.id==body.evidence_id)):
            raise LookupError('证据不属于这场会议')
        ids=[]
        for name,kind in ((body.source_name,body.source_kind),(body.target_name,body.target_kind)):
            command=upsert(k.entities).values(id=uuid.uuid4(),workspace_id=workspace,name=name,kind=kind)
            ids.append(c.scalar(command.on_conflict_do_update(index_elements=['workspace_id','kind','name'],set_={'name':name}).returning(k.entities.c.id)))
        identifier=uuid.uuid4()
        c.execute(insert(k.relations).values(id=identifier,workspace_id=workspace,source_id=ids[0],target_id=ids[1],kind=body.kind,
            session_id=body.session_id,evidence_id=body.evidence_id,valid_from=body.valid_from,valid_to=body.valid_to,active=True))
        bump(c,workspace)
        return identifier

def traverse(c,workspace,names,event_ids,*,hops=2,at=None):
    if not 1<=hops<=3:raise ValueError('关系查询最多三跳')
    at=at or datetime.now(timezone.utc)
    nodes=list(c.execute(select(k.entities).where(k.entities.c.workspace_id==workspace,k.entities.c.name.in_(names))).mappings())
    frontier={node['id'] for node in nodes};seen=set(frontier);edges=[];used=set()
    for _ in range(hops):
        if not frontier:break
        query=select(k.relations).join(k.sessions,k.sessions.c.id==k.relations.c.session_id).where(
            k.relations.c.workspace_id==workspace,k.sessions.c.workspace_id==workspace,k.sessions.c.deleted.is_(False),
            k.relations.c.session_id.in_(event_ids),k.relations.c.active.is_(True),
            or_(k.relations.c.source_id.in_(frontier),k.relations.c.target_id.in_(frontier)),
            or_(k.relations.c.valid_from.is_(None),k.relations.c.valid_from<=at),
            or_(k.relations.c.valid_to.is_(None),k.relations.c.valid_to>at)).limit(1001)
        rows=list(c.execute(query).mappings())
        if len(rows)>1000:raise ValueError('关系查询过大，请缩小范围')
        new=set()
        for row in rows:
            if row['id'] not in used:edges.append(dict(row));used.add(row['id'])
            new.update((row['source_id'],row['target_id']))
        frontier=new-seen;seen.update(new)
    if seen:nodes=list(c.execute(select(k.entities).where(k.entities.c.workspace_id==workspace,k.entities.c.id.in_(seen))).mappings())
    proofs=list(c.execute(select(k.evidence).where(k.evidence.c.workspace_id==workspace,k.evidence.c.id.in_([edge['evidence_id'] for edge in edges]))).mappings()) if edges else []
    return {'nodes':list(map(dict,nodes)),'edges':edges,'evidence':list(map(dict,proofs))}

def alias(repo,workspace,person_id,name):
    with repo.engine.begin() as c:
        if not c.scalar(select(k.people.c.id).where(k.people.c.id==person_id,k.people.c.workspace_id==workspace,k.people.c.active.is_(True))):
            raise LookupError('人物不存在')
        c.execute(upsert(k.aliases).values(id=uuid.uuid4(),person_id=person_id,alias=name,confirmed=True)
            .on_conflict_do_update(index_elements=['person_id','alias'],set_={'confirmed':True}))
        bump(c,workspace)

def remove_alias(repo,workspace,person_id,name):
    with repo.engine.begin() as c:
        if not c.scalar(select(k.people.c.id).where(k.people.c.id==person_id,k.people.c.workspace_id==workspace)):
            raise LookupError('人物不存在')
        c.execute(k.aliases.delete().where(k.aliases.c.person_id==person_id,k.aliases.c.alias==name))
        bump(c,workspace)
