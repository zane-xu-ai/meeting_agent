"""Complete-set SQL tools and deterministic evidence-based answers."""
import json
import time
import uuid
from datetime import datetime,timezone,timedelta
from sqlalchemy import select,func,insert
from knowledge import schema as k
from knowledge.planner import plan,Clarification
from knowledge.store import revision
from persistence.stage_cache import fingerprint
from config import settings


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
    if p.person:
        matches=list(c.execute(select(k.people.c.id,k.people.c.name).outerjoin(k.aliases,k.aliases.c.person_id==k.people.c.id).where(k.people.c.workspace_id==workspace,k.people.c.active.is_(True),
            (k.people.c.name==p.person)|((k.aliases.c.alias==p.person)&k.aliases.c.confirmed.is_(True))).distinct()).mappings())
        if len(matches)>1:raise Clarification('该姓名或别名对应多个人物，请指定身份')
        if matches:p.person=matches[0]['name']
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
        cached=None
        version=revision(c,workspace)
        try:
            p=plan(question,anchor,timezone_name,previous)
            if p.intent=='relations' and not p.terms:raise Clarification('请用引号明确实体名称，例如“上周项目‘甲’的决定”，实体请使用中文双引号')
            normalized=p.model_dump(mode='json')
            cache_key=fingerprint(['query-v2',normalized,version,settings.knowledge_semantic_enabled,settings.knowledge_rerank_enabled,settings.embedding_model,settings.rerank_model])
            if p.intent not in ('search','relations'):
                cached=c.scalar(select(k.queries.c.response).where(k.queries.c.workspace_id==workspace,k.queries.c.knowledge_revision==version,
                    k.queries.c.plan['cache_key'].astext==cache_key,k.queries.c.created_at>datetime.now(timezone.utc)-timedelta(seconds=60)).order_by(k.queries.c.created_at.desc()).limit(1))
            if cached:
                result={**cached,'cache_hit':True}
            else:
                facts,evidence,unknown=execute(c,workspace,p)
                result=respond(p,facts,evidence,unknown)
            normalized.update(cache_key=cache_key,matched_ids=result['matched_ids'])
        except Clarification as error:
            result={'status':'clarify','reply':str(error),'evidence':[],'claims':[]}
            normalized={'question':question}
        if result['status']!='clarify' and p.intent=='relations':
            from knowledge.relations import traverse
            graph=traverse(c,workspace,[p.terms],[event['id'] for event in facts],hops=2,at=anchor)
            labels={node['id']:node['name'] for node in graph['nodes']}
            result.update(graph=graph,evidence=graph['evidence'],claims=[{'evidence_ids':[str(edge['evidence_id'])]} for edge in graph['edges']])
            result['reply']='查询已登记的有效关系：\n'+'\n'.join(labels[edge['source_id']]+' → '+edge['kind']+' → '+labels[edge['target_id']] for edge in graph['edges'])
            result['status']='partial' if graph['edges'] else 'no_match'
        if result['status']!='clarify' and p.intent=='search':
            from knowledge.retrieval import retrieve
            hits,search_coverage=retrieve(c,workspace,[event['id'] for event in facts],p.terms)
            result['retrieval']=search_coverage
            result['status']='partial' if hits else 'no_match'
            result['reply']='搜索结果仅表示命中片段，不能据此推断全部会议的语义总数。\n'+ '\n'.join(hit['text'] for hit in hits)
            result['evidence']=[{'id':str(hit['evidence_id']),'session_id':str(hit['session_id']),'quote':hit['quote'],'kind':hit['kind'],'start_ms':hit['start_ms'],'end_ms':hit['end_ms']} for hit in hits]
            result['matched_ids']=list(dict.fromkeys(str(hit['session_id']) for hit in hits))
            result['coverage']['matched']=len(result['matched_ids'])
            normalized['matched_ids']=result['matched_ids']
        result.update(query_id=str(identifier),plan=normalized,as_of=anchor.isoformat(),timezone=timezone_name)
        result=json_safe(result)
        c.execute(insert(k.queries).values(id=identifier,workspace_id=workspace,conversation=conversation,question=question,
            plan=normalized,response=result,knowledge_revision=revision(c,workspace),duration_ms=int((time.monotonic()-started)*1000)))
    return result
