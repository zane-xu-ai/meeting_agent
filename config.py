"""会议助手智能体 - 配置管理"""

from pathlib import Path
from typing import List

from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic import Field


# 默认模型列表
DEFAULT_ASR_MODELS: List[str] = [
    "paraformer-v2",
    "paraformer-v1",
    "qwen-audio-3.0-asr-flash-filetrans",
    "qwen-audio-3.1-asr-flash-filetrans",
]

DEFAULT_LLM_MODELS: List[str] = [
    "qwen3.7-flash-2026-07-15",
    "qwen3.7-flash",
    "deepseek-v4-flash-0731",
    "qwen3.8-max",
    "glm-5.3",
    "qwen3.8-flash",
    "qwen3.8-max-0902",
    "deepseek-v4.1-flash",
]


class Settings(BaseSettings):
    """应用配置，从 .env 文件和环境变量读取"""

    # 阿里云百炼 DashScope (ASR + LLM 共用)
    dashscope_api_key: str = Field(default="", description="阿里云百炼 API Key")

    # LLM (未单独配置时自动复用 dashscope_api_key)
    llm_api_key: str = Field(
        default="", description="LLM API Key，默认复用 DASHSCOPE_API_KEY")
    llm_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="LLM API Base URL",
    )
    llm_models: List[str] = Field(
        default=DEFAULT_LLM_MODELS, description="LLM 模型列表，按优先级排序")

    # ASR 配置
    asr_models: List[str] = Field(
        default=DEFAULT_ASR_MODELS, description="ASR 模型列表，按优先级排序")
    asr_sample_rate: int = Field(default=16000, description="音频采样率")

    # 文本处理
    max_tokens_per_chunk: int = Field(
        default=6000, description="分段摘要时每个切片的最大 token 数"
    )
    min_sentence_length: int = Field(
        default=10, description="短句合并阈值，低于此长度的连续同说话人片段会合并"
    )

    # M5: Embedding & Rerank
    embedding_model: str = Field(
        default="qwen3.7-text-embedding-flash",
        description="Embedding 模型名称",
    )
    rerank_model: str = Field(
        default="qwen3.7-text-rerank",
        description="Rerank 模型名称",
    )
    vector_store_dir: Path = Field(
        default=Path("./data/vector_store"),
        description="ChromaDB 向量库持久化目录",
    )

    # M10: 多用户配额
    daily_task_limit: int = Field(
        default=5,
        description="每用户每天的任务数量限制"
    )
    quota_whitelist_ips: List[str] = Field(
        default_factory=list,
        description="配额白名单 IP 列表，这些 IP 不限制"
    )
    quota_data_dir: Path = Field(
        default=Path("./data"),
        description="配额数据存储目录"
    )

    # M11: LangSmith 可观测性
    langsmith_api_key: str = Field(
        default="",
        description="LangSmith API Key，用于全链路追踪"
    )
    langchain_project: str = Field(
        default="meeting_agent",
        description="LangChain 项目名称 (LangSmith 中显示)"
    )
    langchain_tracing_v2: bool = Field(
        default=False,
        description="是否启用 LangChain Tracing V2"
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API 端点"
    )

    # 输出
    output_dir: Path = Field(default=Path("./output"), description="输出目录")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @model_validator(mode="after")
    def _fallback_llm_key(self) -> "Settings":
        """LLM 未单独配置 Key 时，自动复用 DashScope Key"""
        if not self.llm_api_key and self.dashscope_api_key:
            self.llm_api_key = self.dashscope_api_key
        return self


settings = Settings()
