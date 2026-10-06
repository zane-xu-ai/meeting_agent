"""
M5: Rerank 客户端 — DashScope text-rerank (OpenAI 兼容接口)

封装 qwen3.7-text-rerank 调用，对候选文档做精排。
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx
from loguru import logger

from config import settings


@dataclass
class RerankResult:
    """Rerank 结果"""
    index: int       # 原始索引 (在输入 documents 中的位置)
    content: str     # 文档内容
    score: float     # 相关性分数


class RerankerClient:
    """DashScope Rerank 客户端"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or settings.rerank_model
        self.api_key = api_key or settings.llm_api_key
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")

        if not self.api_key:
            raise ValueError(
                "Rerank API Key 未配置: 请设置 DASHSCOPE_API_KEY"
            )

        self.rerank_url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            f"Rerank 客户端初始化: model={self.model}, "
            f"base_url={self.base_url}"
        )

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
    ) -> list[RerankResult]:
        """
        对候选文档重排序

        Args:
            query: 查询文本
            documents: 候选文档列表
            top_n: 返回前 N 个

        Returns:
            按相关性降序排列的结果列表
        """
        if not documents:
            logger.debug("[Rerank] 候选文档为空，跳过")
            return []

        logger.debug(
            f"[Rerank] 开始: query={query[:50]}..., "
            f"候选 {len(documents)} 篇, top_n={top_n}"
        )

        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "top_n": min(top_n, len(documents)),
                "return_documents": False,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(
                    self.rerank_url, json=payload, headers=self._headers
                )
                resp.raise_for_status()
                data = resp.json()

                # DashScope rerank 返回格式:
                # {"results": [{"index": 0, "relevance_score": 0.95, "document": {"text": "..."}}]}
                results: list[RerankResult] = []
                for item in data.get("results", []):
                    idx = item["index"]
                    score = item["relevance_score"]
                    content = documents[idx] if idx < len(documents) else ""
                    results.append(
                        RerankResult(index=idx, content=content, score=score)
                    )

                # 按分数降序排序
                results.sort(key=lambda x: x.score, reverse=True)

                logger.info(
                    f"[Rerank] 完成: {len(documents)} 候选 → "
                    f"top-{len(results)}, "
                    f"最高分={results[0].score:.4f}" if results else "无结果"
                )

                return results

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"[Rerank] HTTP 错误: {e.response.status_code} - "
                    f"{e.response.text[:200]}"
                )
                raise
            except Exception as e:
                logger.error(f"[Rerank] 调用失败: {e}")
                raise


# 全局单例
_reranker_client: RerankerClient | None = None


def get_reranker_client() -> RerankerClient:
    """获取全局 RerankerClient 实例"""
    global _reranker_client
    if _reranker_client is None:
        _reranker_client = RerankerClient()
    return _reranker_client
