"""会议助手智能体 - 配置管理"""

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic import Field


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
    llm_model: str = Field(default="qwen-plus", description="LLM 模型名称")

    # ASR 配置
    asr_model: str = Field(default="paraformer-v2", description="ASR 模型名称")
    asr_sample_rate: int = Field(default=16000, description="音频采样率")

    # 文本处理
    max_tokens_per_chunk: int = Field(
        default=6000, description="分段摘要时每个切片的最大 token 数"
    )
    min_sentence_length: int = Field(
        default=10, description="短句合并阈值，低于此长度的连续同说话人片段会合并"
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
