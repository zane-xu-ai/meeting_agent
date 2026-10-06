"""
M5: ChromaDB 向量存储 — 父子块双 Collection 架构

设计:
- child collection: 存储子块 + embedding，用于向量检索 (高召回)
- parent collection: 存储父块内容 (无 embedding)，用于检索后取回完整上下文

检索流程:
1. 子块向量粗检索 (top-20)
2. Rerank 精排 (top-5)
3. 取回对应父块作为上下文
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from agent.chunker import ParentChunk
from config import settings


@dataclass
class SearchResult:
    """检索结果"""
    child_content: str      # 命中的子块
    parent_content: str     # 对应的父块 (注入 LLM 的上下文)
    score: float            # rerank 分数 (或 embedding 相似度)
    task_id: str            # 任务 ID
    source_type: str        # "summary" | "asr"
    source_file: str        # 原始文件名
    section: str            # 所属章节
    parent_id: str          # 父块 ID


class VectorStore:
    """ChromaDB 向量存储 (父子块架构)"""

    def __init__(
        self,
        persist_dir: Path | None = None,
        child_collection: str = "meeting_child",
        parent_collection: str = "meeting_parent",
    ):
        """
        初始化向量存储

        Args:
            persist_dir: ChromaDB 持久化目录
            child_collection: 子块集合名称
            parent_collection: 父块集合名称
        """
        try:
            import chromadb
        except ImportError:
            logger.error(
                "[VectorStore] chromadb 未安装，请执行: pip install chromadb"
            )
            raise

        self.persist_dir = persist_dir or settings.vector_store_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[VectorStore] 初始化: persist_dir={self.persist_dir}, "
            f"child={child_collection}, parent={parent_collection}"
        )

        # ChromaDB 持久化客户端
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        # 双 Collection
        self.child_col = self.client.get_or_create_collection(
            name=child_collection,
            metadata={"hnsw:space": "cosine"},  # 余弦相似度
        )
        self.parent_col = self.client.get_or_create_collection(
            name=parent_collection,
        )

        logger.info(
            f"[VectorStore] Collection 就绪: "
            f"child={self.child_col.count()} 条, "
            f"parent={self.parent_col.count()} 条"
        )

    async def add_chunks(
        self,
        parent_chunks: list[ParentChunk],
        child_embeddings: list[list[float]],
    ) -> int:
        """
        写入父子块

        Args:
            parent_chunks: 父块列表 (每个父块包含子块列表)
            child_embeddings: 所有子块的 embedding (顺序与子块展开顺序一致)

        Returns:
            写入的子块数量
        """
        if not parent_chunks:
            logger.debug("[VectorStore] 无父块可写入")
            return 0

        # 展平所有子块
        all_children = []
        for parent in parent_chunks:
            for child in parent.children:
                all_children.append((parent, child))

        if len(all_children) != len(child_embeddings):
            logger.error(
                f"[VectorStore] 子块数量不匹配: "
                f"{len(all_children)} vs {len(child_embeddings)}"
            )
            raise ValueError("子块与 embedding 数量不一致")

        logger.debug(
            f"[VectorStore] 开始写入: {len(parent_chunks)} 父块, "
            f"{len(all_children)} 子块"
        )

        # 写入父块 (无 embedding)
        parent_ids = []
        parent_docs = []
        parent_metas = []
        for parent in parent_chunks:
            parent_ids.append(parent.parent_id)
            parent_docs.append(parent.content[:500])  # ChromaDB 需要 document
            parent_metas.append({
                "task_id": parent.task_id,
                "rec_type": parent.rec_type,
                "source_type": parent.source_type,
                "source_file": parent.source_file,
                "created_at": parent.created_at,
                "section": parent.section,
                "content": parent.content,  # 完整内容存 metadata
            })

        # 分批写入 (ChromaDB 限制)
        batch_size = 100
        for i in range(0, len(parent_ids), batch_size):
            self.parent_col.add(
                ids=parent_ids[i : i + batch_size],
                documents=parent_docs[i : i + batch_size],
                metadatas=parent_metas[i : i + batch_size],
            )

        logger.debug(
            f"[VectorStore] 父块写入完成: {len(parent_ids)} 条"
        )

        # 写入子块 (有 embedding)
        child_ids = []
        child_docs = []
        child_metas = []
        child_embs = []

        for idx, (parent, child) in enumerate(all_children):
            child_ids.append(child.child_id)
            child_docs.append(child.content)
            child_metas.append({
                "parent_id": child.parent_id,
                "task_id": parent.task_id,
                "rec_type": parent.rec_type,
                "source_type": parent.source_type,
                "source_file": parent.source_file,
                "section": parent.section,
                "chunk_index": child.chunk_index,
            })
            child_embs.append(child_embeddings[idx])

        for i in range(0, len(child_ids), batch_size):
            self.child_col.add(
                ids=child_ids[i : i + batch_size],
                documents=child_docs[i : i + batch_size],
                metadatas=child_metas[i : i + batch_size],
                embeddings=child_embs[i : i + batch_size],
            )

        logger.info(
            f"[VectorStore] 写入完成: {len(parent_ids)} 父块, "
            f"{len(child_ids)} 子块"
        )
        return len(child_ids)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[tuple[str, dict, float]]:
        """
        子块向量检索 (粗检索)

        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            filters: 元数据过滤 (如 {"source_type": "summary"})

        Returns:
            [(child_content, metadata, score), ...]
        """
        where_filter = None
        if filters:
            # ChromaDB where 语法
            where_filter = filters

        results = self.child_col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.child_col.count() or 1),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # 解析结果
        hits = []
        if results and results["ids"] and results["ids"][0]:
            for i, child_id in enumerate(results["ids"][0]):
                doc = results["documents"][0][i] if results["documents"] else ""
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0.0
                # cosine distance → similarity (1 - distance)
                score = 1.0 - dist
                hits.append((doc, meta, score))

        logger.debug(
            f"[VectorStore] 粗检索: top_k={top_k}, 命中 {len(hits)} 条"
        )
        return hits

    async def get_parent(self, parent_id: str) -> dict | None:
        """
        根据 parent_id 取回父块

        Args:
            parent_id: 父块 ID

        Returns:
            父块元数据 (包含完整 content) 或 None
        """
        try:
            result = self.parent_col.get(ids=[parent_id])
            if result and result["ids"]:
                meta = result["metadatas"][0] if result["metadatas"] else {}
                logger.debug(
                    f"[VectorStore] 取回父块: {parent_id}, "
                    f"content 长度={len(meta.get('content', ''))}"
                )
                return meta
            else:
                logger.debug(f"[VectorStore] 父块未找到: {parent_id}")
                return None
        except Exception as e:
            logger.error(f"[VectorStore] 取回父块失败: {e}")
            return None

    async def search_with_rerank(
        self,
        query: str,
        query_embedding: list[float],
        reranker,
        coarse_top_k: int = 20,
        final_top_k: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """
        三阶段检索: 粗检索 → Rerank 精排 → 取回父块

        Args:
            query: 查询文本
            query_embedding: 查询向量
            reranker: RerankerClient 实例
            coarse_top_k: 粗检索数量
            final_top_k: 精排后数量
            filters: 元数据过滤

        Returns:
            检索结果列表 (包含父块上下文)
        """
        # 阶段 1: 粗检索
        coarse_hits = await self.search(query_embedding, coarse_top_k, filters)
        if not coarse_hits:
            logger.info("[VectorStore] 粗检索无结果")
            return []

        logger.debug(
            f"[VectorStore] 粗检索: {len(coarse_hits)} 条候选"
        )

        # 阶段 2: Rerank 精排
        candidate_docs = [hit[0] for hit in coarse_hits]
        rerank_results = await reranker.rerank(query, candidate_docs, final_top_k)

        if not rerank_results:
            logger.warning("[VectorStore] Rerank 无结果，使用粗检索结果")
            rerank_results = [
                type('obj', (object,), {'index': i, 'content': doc, 'score': score})
                for i, (doc, _, score) in enumerate(coarse_hits[:final_top_k])
            ]

        # 阶段 3: 取回父块
        final_results = []
        for rr in rerank_results:
            original_idx = rr.index
            if original_idx >= len(coarse_hits):
                continue

            child_content, meta, _ = coarse_hits[original_idx]
            parent_id = meta.get("parent_id", "")

            # 取回父块
            parent_meta = await self.get_parent(parent_id)
            parent_content = parent_meta.get("content", "") if parent_meta else ""

            result = SearchResult(
                child_content=child_content,
                parent_content=parent_content,
                score=rr.score,
                task_id=meta.get("task_id", ""),
                source_type=meta.get("source_type", ""),
                source_file=meta.get("source_file", ""),
                section=meta.get("section", ""),
                parent_id=parent_id,
            )
            final_results.append(result)

        logger.info(
            f"[VectorStore] 三阶段检索完成: "
            f"粗检索 {len(coarse_hits)} → Rerank top-{len(final_results)}"
        )
        return final_results

    async def delete_by_task_id(self, task_id: str) -> int:
        """
        删除指定任务的所有向量

        Args:
            task_id: 任务 ID

        Returns:
            删除的子块数量
        """
        # 查找该任务的所有子块
        results = self.child_col.get(
            where={"task_id": task_id},
            include=[],
        )

        if not results or not results["ids"]:
            logger.debug(f"[VectorStore] 任务 {task_id} 无向量可删")
            return 0

        child_ids = results["ids"]

        # 查找关联的父块
        parent_ids = set()
        for meta in results.get("metadatas", []):
            if meta and "parent_id" in meta:
                parent_ids.add(meta["parent_id"])

        # 删除子块
        self.child_col.delete(ids=child_ids)

        # 删除父块
        if parent_ids:
            self.parent_col.delete(ids=list(parent_ids))

        logger.info(
            f"[VectorStore] 删除任务 {task_id}: "
            f"{len(child_ids)} 子块, {len(parent_ids)} 父块"
        )
        return len(child_ids)

    async def delete_by_source_file(self, source_file: str) -> int:
        """
        删除指定源文件的所有向量 (防止重复索引)

        Args:
            source_file: 源文件名 (如 "05-09 内部会议_paraformer-v2.txt")

        Returns:
            删除的子块数量
        """
        results = self.child_col.get(
            where={"source_file": source_file},
            include=["metadatas"],
        )

        if not results or not results.get("ids"):
            return 0

        child_ids = results["ids"]

        # 查找关联的父块
        parent_ids = set()
        for meta in (results.get("metadatas") or []):
            if meta and "parent_id" in meta:
                parent_ids.add(meta["parent_id"])

        # 删除子块
        self.child_col.delete(ids=child_ids)

        # 删除父块
        if parent_ids:
            self.parent_col.delete(ids=list(parent_ids))

        logger.info(
            f"[VectorStore] 删除源文件 {source_file}: "
            f"{len(child_ids)} 子块, {len(parent_ids)} 父块"
        )
        return len(child_ids)

    def get_stats(self) -> dict:
        """
        统计信息

        Returns:
            {"child_count": int, "parent_count": int, "persist_dir": str}
        """
        return {
            "child_count": self.child_col.count(),
            "parent_count": self.parent_col.count(),
            "persist_dir": str(self.persist_dir),
        }


# 全局单例
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """获取全局 VectorStore 实例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
