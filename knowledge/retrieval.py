"""Authorized lexical/cosine search with explicit projection coverage."""
import asyncio
import math
from sqlalchemy import select,update
from knowledge import schema as k
from config import settings


def cosine(left,right):
    if len(left)!=len(right) or not left:return 0.0
    norm=math.sqrt(sum(v*v for v in left)*sum(v*v for v in right))
    return sum(a*b for a,b in zip(left,right))/norm if norm else 0.0

def index(repo,embedder=None):
    if embedder is None:
        if not settings.knowledge_semantic_enabled:return 0
        from agent.embedding import EmbeddingClient
        embedder=EmbeddingClient()
    with repo.engine.connect() as c:
        rows=list(c.execute(select(k.chunks).join(k.sessions,k.sessions.c.id==k.chunks.c.session_id).where(
            k.sessions.c.deleted.is_(False),k.chunks.c.source_revision==k.sessions.c.revision,
            (k.chunks.c.embedding.is_(None))|(k.chunks.c.embedding_model!=embedder.model)).limit(100)).mappings())
    if not rows:return 0
    vectors=asyncio.run(embedder.embed_batch([row['text'] for row in rows]))
    if len(vectors)!=len(rows):raise ValueError('向量返回数量不符')
    with repo.engine.begin() as c:
        for row,vector in zip(rows,vectors):
            if not vector or any(not math.isfinite(float(v)) for v in vector):raise ValueError('无效向量')
            c.execute(update(k.chunks).where(k.chunks.c.id==row['id'],k.chunks.c.source_revision==row['source_revision'])
                      .values(embedding=vector,embedding_model=embedder.model))
    return len(rows)

def retrieve(c,workspace,event_ids,terms,*,embedder=None,reranker=None,limit=20):
    query=select(k.chunks,k.evidence.c.quote,k.evidence.c.kind,k.evidence.c.analysis_id,k.evidence.c.start_ms,k.evidence.c.end_ms)
    query=query.join(k.sessions,k.chunks.c.session_id==k.sessions.c.id).join(k.evidence,k.chunks.c.evidence_id==k.evidence.c.id)
    rows=list(c.execute(query.where(k.chunks.c.workspace_id==workspace,k.chunks.c.session_id.in_(event_ids),
        k.sessions.c.workspace_id==workspace,k.sessions.c.deleted.is_(False),k.chunks.c.source_revision==k.sessions.c.revision)
        .order_by(k.chunks.c.id).limit(20001)).mappings())
    truncated=len(rows)>20000;rows=rows[:20000]
    semantic_error=None
    if embedder is None and settings.knowledge_semantic_enabled:
        from agent.embedding import EmbeddingClient
        embedder=EmbeddingClient()
    vector=None
    if embedder:
        try:vector=asyncio.run(embedder.embed_text(terms))
        except Exception:semantic_error='embedding_unavailable'
    lexical=sorted((row for row in rows if terms.casefold() in row['text'].casefold()),key=lambda row:row['text'].casefold().count(terms.casefold()),reverse=True)
    semantic=[]
    if vector is not None:
        semantic=sorted((row for row in rows if row['embedding'] and row['embedding_model']==embedder.model),
                        key=lambda row:cosine(vector,row['embedding']),reverse=True)[:limit]
    ranks={}
    for group in (lexical,semantic):
        for position,row in enumerate(group[:limit]):
            item=ranks.setdefault(str(row['id']),{'row':row,'score':0})
            item['score']+=1/(60+position)
    ranked=sorted(ranks.values(),key=lambda item:item['score'],reverse=True)
    if reranker is None and settings.knowledge_rerank_enabled and ranked:
        from agent.reranker import RerankerClient
        reranker=RerankerClient()
    if reranker and ranked:
        try:
            result=asyncio.run(reranker.rerank(terms,[item['row']['text'] for item in ranked],top_n=limit))
            ranked=[ranked[item.index] for item in result if 0<=item.index<len(ranked)]
        except Exception:semantic_error='rerank_unavailable'
    chosen=[dict(item['row']) for item in ranked[:limit]]
    return chosen,{'scanned_chunks':len(rows),'truncated':truncated or len(lexical)>limit,
                   'indexed_chunks':sum(bool(row['embedding']) for row in rows),'semantic':vector is not None,
                   'fallback_reason':semantic_error,'exhaustive':False}
