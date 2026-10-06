"""M11: LangSmith 可观测性配置

通过 LangSmith 实现全链路追踪，记录每个 LangGraph 节点的:
- 输入/输出
- 执行耗时
- Token 用量
- 错误信息

配置方式 (环境变量或 .env):
    LANGCHAIN_TRACING_V2=true
    LANGSMITH_API_KEY=lsv2_pt_xxx
    LANGCHAIN_PROJECT=meeting_agent

启用后，所有 LangGraph 执行轨迹自动上传到 LangSmith 控制台:
    https://smith.langchain.com/
"""

import os
from loguru import logger

from config import settings


def setup_langsmith_tracing():
    """初始化 LangSmith 追踪

    通过设置环境变量启用 LangChain/LangGraph 的内置追踪。
    LangSmith SDK 会自动拦截并上传执行轨迹。
    """
    # 检查是否已配置
    if not settings.langsmith_api_key:
        logger.info("[M11 Tracing] LANGSMITH_API_KEY 未配置，追踪功能已禁用")
        return False

    if not settings.langchain_tracing_v2:
        logger.info("[M11 Tracing] LANGCHAIN_TRACING_V2=false，追踪功能已禁用")
        return False

    # 设置环境变量 (LangSmith SDK 读取这些变量)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    logger.info(
        f"[M11 Tracing] LangSmith 追踪已启用 | "
        f"project={settings.langchain_project} | "
        f"endpoint={settings.langsmith_endpoint}"
    )

    # 验证 langsmith 包是否安装
    try:
        import langsmith
        logger.debug(f"[M11 Tracing] langsmith v{langsmith.__version__} 已安装")
    except ImportError:
        logger.warning(
            "[M11 Tracing] langsmith 包未安装，追踪功能可能无法正常工作。"
            "请运行: pip install langsmith"
        )

    return True


def get_tracing_info() -> dict:
    """获取当前追踪配置信息"""
    return {
        "enabled": settings.langchain_tracing_v2 and bool(settings.langsmith_api_key),
        "project": settings.langchain_project,
        "endpoint": settings.langsmith_endpoint,
        "api_key_configured": bool(settings.langsmith_api_key),
    }


# 模块加载时自动初始化
_tracing_enabled = setup_langsmith_tracing()
