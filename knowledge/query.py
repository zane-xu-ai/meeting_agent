"""Complete-set SQL tools and deterministic evidence-based answers."""
import json
import time
import uuid
from datetime import datetime,timezone
from sqlalchemy import select,func,insert
from knowledge import schema as k
from knowledge.planner import plan,Clarification
from knowledge.store import revision


def json_safe(value):
    return json.loads(json.dumps(value,ensure_ascii=False,default=lambda item:item.isoformat() if hasattr(item,'isoformat') else str(item)))

def execute(c,workspace,p):
    base=(k.sessions.c.workspace_id==workspace,k.sessions.c.deleted.is_(False),k.sessions.c.category==p.category)
    unknown=c.scalar(select(func.count()).select_from(k.sessions).where(*base,k.sessions.c.time_status!='confirmed'))
    query=select(k.sessions).where(*base,k.sessions.c.time_status=='confirmed',k.sessions.c.event_date>=p.start,k.sessions.c.event_date<p.stop)
    if p.inherited_ids is not None:query=query.where(k.sessions.c.id.in_([uuid.UUID(value) for value in p.inherited_ids]))
    events=list(c.execute(query.order_by(k.sessions.c.event_date,k.sessions.c.id)).mappings())
    ids=[item['id'] for item in events]
    if len(ids)>10000:raise Clarification('匹配会议超过一万场，请缩小范围')
    participants=list(c.execute(select(k.participants).where(k.participants.c.session_id.in_(ids),k.participants.c.active.is_(True))).mappings()) if ids else []
    topics=list(c.execute(select(k.topics).where(k.topics.c.session_id.in_(ids),k.topics.c.active.is_(True))).mappings()) if ids else []
    evidence_ids={row['evidence_id'] for row in [*participants,*topics]}
    proofs=list(c.execute(select(k.evidence).where(k.evidence.c.workspace_id==workspace,k.evidence.c.id.in_(evidence_ids))).mappings()) if evidence_ids else []
    facts=[]
    for event in events:
        attendance=[dict(item) for item in participants if item['session_id']==event['id']]
        subjects=[dict(item) for item in topics if item['session_id']==event['id']]
        facts.append({'id':event['id'],'title':event['title'],'date':event['event_date'],
                      'participants':attendance,'topics':subjects})
    return facts,list(map(dict,proofs)),unknown

def respond(p,facts,evidence,unknown):
    matched=facts
    missing=0
    lines=[f'查询范围 {p.start} 至 {p.stop}（结束日期不包含），类别 {p.category}。']
    claims=[]
    if p.intent=='search':
        # Exact lexical filtering is explicit in baseline, not a claim of semantic recall.
        matched=[event for event in facts if any(p.terms in item['title'] or p.terms in item['summary'] for item in event['topics'])]
        lines.append(f'已扫描 {len(facts)} 场会议的已提取议题，关键词“{p.terms}”命中 {len(matched)} 场。')
        missing=sum(not event['topics'] for event in facts)
    elif p.intent=='count':lines.append(f'按会议事件去重，共 {len(facts)} 场。')
    elif p.intent=='attendance_count':
        people={}
        for event in facts:
            seen=set()
            for row in event['participants']:
                if row['status']!='confirmed' or not row['person_id']:continue
                if p.person and p.person!=row['display_name']:continue
                person=str(row['person_id'])
                if person in seen:continue
                seen.add(person)
                entry=people.setdefault(person,{'name':row['display_name'],'count':0,'evidence_ids':[]})
                entry['count']+=1;entry['evidence_ids'].append(str(row['evidence_id']))
            if not any(row['status']=='confirmed' for row in event['participants']):missing+=1
        claims=sorted(people.values(),key=lambda item:(-item['count'],item['name']))
        lines.extend(f"{item['name']}：确认参会 {item['count']} 次。" for item in claims)
        if not claims:lines.append('没有满足条件的已确认参会关系。')
    for event in matched:
        if p.intent=='attendance_count':continue
        lines.append(f"\n{event['date']} · {event['title']}")
        if p.intent in ('attendees','speakers'):
            status='observed_speaker' if p.intent=='speakers' else 'confirmed'
            rows=[row for row in event['participants'] if row['status']==status]
            if not rows:
                missing+=1
                observed=[row['display_name'] for row in event['participants'] if row['status']=='observed_speaker']
                lines.append('参会名单尚未确认。'+('观察到发言标签：'+'、'.join(observed) if observed else ''))
            else:
                lines.append(('观察到发言者：' if p.intent=='speakers' else '确认参会：')+'、'.join(dict.fromkeys(row['display_name'] for row in rows)))
                claims.append({'session_id':str(event['id']),'names':list(dict.fromkeys(row['display_name'] for row in rows)),
                               'evidence_ids':[str(row['evidence_id']) for row in rows]})
        elif p.intent in ('topics','search'):
            if not event['topics']:missing+=1;lines.append('议题尚未提取。')
            for item in event['topics']:
                lines.append(f"- {item['title']}：{item['summary']}")
                claims.append({'session_id':str(event['id']),'text':item['summary'],'evidence_ids':[str(item['evidence_id'])]})
    if not facts:lines.append('没有匹配的已确认日期会议。')
    if unknown:lines.append(f'另有 {unknown} 场同类记录的日期未知或待确认，尚不能判定是否在此范围。')
    if missing:lines.append(f'匹配范围中有 {missing} 场所需事实不完整。')
    state='partial' if missing or unknown else 'complete' if facts else 'no_match'
    return {'status':state,'reply':'\n'.join(lines),'claims':claims,'evidence':evidence,
            'coverage':{'scanned':len(facts),'matched':len(matched),'unknown_dates':unknown,'missing_facts':missing},
            'matched_ids':[str(event['id']) for event in matched]}

def query(repo,workspace,question,conversation='default',*,anchor=None,timezone_name='Asia/Shanghai'):
    started=time.monotonic();identifier=uuid.uuid4()
    anchor=anchor or datetime.now(timezone.utc)
    with repo.engine.begin() as c:
        previous=c.scalar(select(k.queries.c.plan).where(k.queries.c.workspace_id==workspace,k.queries.c.conversation==conversation)
                          .order_by(k.queries.c.created_at.desc(),k.queries.c.id.desc()).limit(1))
        try:
            p=plan(question,anchor,timezone_name,previous)
            if p.intent=='relations':raise Clarification('请先指定项目、决策或行动项，关系查询需要已登记的证据')
            facts,evidence,unknown=execute(c,workspace,p)
            result=respond(p,facts,evidence,unknown)
            normalized=p.model_dump(mode='json')
            normalized['matched_ids']=result['matched_ids']
        except Clarification as error:
            result={'status':'clarify','reply':str(error),'evidence':[],'claims':[]}
            normalized={'question':question}
        result.update(query_id=str(identifier),plan=normalized,as_of=anchor.isoformat(),timezone=timezone_name)
        result=json_safe(result)
        c.execute(insert(k.queries).values(id=identifier,workspace_id=workspace,conversation=conversation,question=question,
            plan=normalized,response=result,knowledge_revision=revision(c,workspace),duration_ms=int((time.monotonic()-started)*1000)))
    return result
