"""
M5: Embedding 客户端 — DashScope text-embedding (OpenAI 兼容接口)

封装 qwen3.7-text-embedding-flash 调用，支持单条和批量文本向量化。
"""
from __future__ import annotations

import httpx
from loguru import logger

from config import settings


class EmbeddingClient:
    """DashScope Embedding 客户端 (OpenAI 兼容接口)"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or settings.embedding_model
        self.api_key = api_key or settings.llm_api_key
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")

        if not self.api_key:
            raise ValueError(
                "Embedding API Key 未配置: 请设置 DASHSCOPE_API_KEY"
            )

        self.embed_url = f"{self.base_url}/embeddings"
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            f"Embedding 客户端初始化: model={self.model}, "
            f"base_url={self.base_url}"
        )

    async def embed_text(self, text: str) -> list[float]:
        """
        单条文本 → 向量

        Args:
            text: 待向量化文本

        Returns:
            浮点数向量 (1024 维)
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(
        self, texts: list[str], batch_size: int = 25
    ) -> list[list[float]]:
        """
        批量文本 → 向量列表 (自动分批)

        DashScope 限制每次最多 25 条，超过自动分批。

        Args:
            texts: 待向量化文本列表
            batch_size: 每批大小 (默认 25)

        Returns:
            向量列表，顺序与输入一致
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        n_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_idx = i // batch_size + 1

            logger.debug(
                f"[Embedding] 批次 {batch_idx}/{n_batches}: "
                f"{len(batch)} 条文本"
            )

            embeddings = await self._embed_batch_single(batch)
            all_embeddings.extend(embeddings)

        logger.info(
            f"[Embedding] 完成: {len(texts)} 条文本 → "
            f"{len(all_embeddings)} 个向量, 模型={self.model}"
        )
        return all_embeddings

    async def _embed_batch_single(self, texts: list[str]) -> list[list[float]]:
        """
        单批 Embedding 调用

        Args:
            texts: 文本列表 (<= 25 条)

        Returns:
            向量列表
        """
        payload = {
            "model": self.model,
            "input": texts,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(
                    self.embed_url, json=payload, headers=self._headers
                )
                resp.raise_for_status()
                data = resp.json()

                # OpenAI 兼容格式: {"data": [{"embedding": [...], "index": 0}, ...]}
                embeddings = [
                    item["embedding"]
                    for item in sorted(data["data"], key=lambda x: x["index"])
                ]

                usage = data.get("usage", {})
                logger.debug(
                    f"[Embedding] 批次成功: {len(embeddings)} 个向量, "
                    f"tokens={usage.get('total_tokens', '?')}"
                )
                return embeddings

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"[Embedding] HTTP 错误: {e.response.status_code} - "
                    f"{e.response.text[:200]}"
                )
                raise
            except Exception as e:
                logger.error(f"[Embedding] 调用失败: {e}")
                raise


# 全局单例
_embedding_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    """获取全局 EmbeddingClient 实例"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client
