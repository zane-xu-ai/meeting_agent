"""会议助手 - 录音类型分类器

根据 ASR 转写文本判断录音类型: meeting / interview / general
"""

from __future__ import annotations

from loguru import logger

from agent.llm_client import LLMClient

CLASSIFY_PROMPT = """你是一个录音类型分类器。请根据以下转写文本，判断录音属于哪种类型。

## 类型定义

1. **interview** (面试录音): 包含面试官提问和面试者回答的场景，通常有自我介绍、技术问答、项目经验讨论等
2. **meeting** (会议录音): 多人参与的工作会议，有议题讨论、决策、任务分配等特征
3. **general** (其他录音): 普通对话、讲座、访谈(非面试)、日常交流等不属于以上两类的录音

## 判断依据

- 面试: 存在明显的"提问-回答"模式，一方提问另一方作答，有技术/经验考察意图
- 会议: 多人讨论，有议题、决策、待办分工
- 其他: 不符合以上特征

## 输出格式

只输出一个 JSON，不要输出其他内容：
```json
{{
  "type": "interview|meeting|general",
  "confidence": 0.0-1.0,
  "reason": "简要说明判断理由"
}}
```

## 转写文本

{transcript}
"""


async def classify_recording(transcript_text: str) -> str:
    """
    根据转写文本判断录音类型。

    Returns:
        "meeting" | "interview" | "general"
    """
    # 截取前 3000 字符用于分类(避免 token 浪费)
    sample = transcript_text[:3000]

    llm = LLMClient()
    prompt = CLASSIFY_PROMPT.format(transcript=sample)

    try:
        data = await llm.chat_json(prompt)
        rec_type = data.get("type", "general")
        confidence = data.get("confidence", 0.0)
        reason = data.get("reason", "")

        # 校验类型值
        if rec_type not in ("meeting", "interview", "general"):
            logger.warning(f"分类结果无效: {rec_type}，默认使用 general")
            rec_type = "general"

        logger.info(
            f"录音分类: {rec_type} (confidence={confidence}, reason={reason})")
        return rec_type

    except Exception as e:
        logger.warning(f"录音分类失败: {e}，默认使用 general")
        return "general"
