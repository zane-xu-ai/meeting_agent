"""
M3: Context Manager — 上下文工程模块

负责:
- Token 精确计算 (tiktoken)
- 上下文截断与压缩
- 动态 system prompt 组装 (按 token 预算分配)
- 长文本检测 (是否需要摘要)
"""
from __future__ import annotations

import tiktoken
from loguru import logger


class ContextManager:
    """
    上下文管理器

    职责:
    1. 精确计算文本的 token 数
    2. 按 token 预算截断文本
    3. 动态组装 system prompt (base + transcript + analysis)
    4. 检测文本是否超出上下文窗口
    """

    def __init__(
        self,
        max_context_tokens: int = 120_000,
        encoding_name: str = "cl100k_base",
    ):
        """
        Args:
            max_context_tokens: 模型最大上下文 token 数
                qwen-plus: 131,072 tokens
                qwen-turbo: 1,000,000 tokens
                默认取较小值以保证兼容性
            encoding_name: tiktoken 编码器名称
                cl100k_base 适用于 GPT-4/Qwen 系列 (近似)
        """
        self.max_context_tokens = max_context_tokens
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
            logger.info(f"ContextManager: 使用 {encoding_name} 编码器")
        except Exception as e:
            logger.warning(
                f"ContextManager: 无法加载 {encoding_name}, 使用字符估算: {e}")
            self._encoding = None

    # ========== Token 计算 ==========

    def count_tokens(self, text: str) -> int:
        """
        计算文本的 token 数

        Args:
            text: 待计算文本

        Returns:
            token 数 (精确或估算)
        """
        if not text:
            return 0
        if self._encoding:
            return len(self._encoding.encode(text))
        # 降级: 按字符估算 (中文 ~1.5 字/token, 英文 ~4 字符/token)
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def count_messages_tokens(self, messages: list[dict]) -> int:
        """
        计算 messages 列表的总 token 数

        包含每条消息的 role + content 开销 (约 4 tokens/条)

        Args:
            messages: [{"role": "...", "content": "..."}, ...]

        Returns:
            总 token 数
        """
        total = 0
        for msg in messages:
            total += 4  # role + 格式开销
            total += self.count_tokens(msg.get("content", ""))
        return total

    # ========== 文本截断 ==========

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        将文本截断到指定 token 数以内

        Args:
            text: 原始文本
            max_tokens: 最大 token 数

        Returns:
            截断后的文本 (末尾添加省略号提示)
        """
        if not text or max_tokens <= 0:
            return text

        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text

        # 按 token 截断
        if self._encoding:
            encoded = self._encoding.encode(text)
            truncated_encoded = encoded[:max_tokens]
            truncated_text = self._encoding.decode(truncated_encoded)
        else:
            # 降级: 按字符比例截断
            ratio = max_tokens / current_tokens
            char_limit = int(len(text) * ratio)
            truncated_text = text[:char_limit]

        logger.info(
            f"文本截断: {current_tokens} tokens → {max_tokens} tokens "
            f"({len(text)} → {len(truncated_text)} 字符)"
        )
        return truncated_text + "\n\n[... 内容过长，已截断 ...]"

    # ========== 上下文组装 ==========

    def build_chat_context(
        self,
        base_system_prompt: str,
        transcript: str,
        analysis: str,
        history: list[dict],
        response_reserve: int = 4000,
    ) -> list[dict]:
        """
        构建多轮对话的完整 messages

        Token 预算分配:
        1. 预留 response_reserve 给模型回复
        2. 计算 history 的 token 数
        3. 剩余预算分配给 transcript 和 analysis (2:1 比例)

        Args:
            base_system_prompt: 基础系统提示词 (不含 transcript/analysis)
            transcript: 转写文本
            analysis: 分析报告
            history: 对话历史 [{"role": "user/assistant", "content": "..."}]
            response_reserve: 为模型回复预留的 token 数

        Returns:
            完整的 messages 列表
        """
        # 1. 计算可用 token 预算
        history_tokens = self.count_messages_tokens(history)
        base_prompt_tokens = self.count_tokens(base_system_prompt)

        available = (
            self.max_context_tokens
            - base_prompt_tokens
            - history_tokens
            - response_reserve
        )

        if available < 1000:
            logger.warning(
                f"上下文预算不足: available={available}, "
                f"history={history_tokens}, base={base_prompt_tokens}"
            )
            available = 1000

        # 2. 分配预算: transcript 占 2/3, analysis 占 1/3
        transcript_budget = int(available * 2 / 3)
        analysis_budget = int(available * 1 / 3)

        logger.info(
            f"上下文预算分配: "
            f"total={self.max_context_tokens}, "
            f"available={available}, "
            f"transcript_budget={transcript_budget}, "
            f"analysis_budget={analysis_budget}"
        )

        # 3. 截断 transcript 和 analysis
        truncated_transcript = self.truncate_to_tokens(
            transcript, transcript_budget)
        truncated_analysis = self.truncate_to_tokens(analysis, analysis_budget)

        # 4. 组装 system prompt
        system_content = base_system_prompt
        if truncated_transcript:
            system_content += f"\n\n## 会议转写文本\n{truncated_transcript}"
        if truncated_analysis:
            system_content += f"\n\n## 分析报告\n{truncated_analysis}"

        # 5. 构建完整 messages
        messages = [{"role": "system", "content": system_content}]
        messages.extend(history)

        total_tokens = self.count_messages_tokens(messages)
        logger.info(f"上下文构建完成: 总计 {total_tokens} tokens")

        return messages

    # ========== 长文本检测 ==========

    def needs_summarization(self, text: str, threshold_tokens: int = 50_000) -> bool:
        """
        检测文本是否需要摘要处理

        当转写文本超过阈值时，应先生成摘要再注入上下文

        Args:
            text: 待检测文本
            threshold_tokens: 阈值 token 数 (默认 50k)

        Returns:
            True 表示需要摘要
        """
        return self.count_tokens(text) > threshold_tokens

    def estimate_text_stats(self, text: str) -> dict:
        """
        估算文本统计信息

        Args:
            text: 待分析文本

        Returns:
            {"chars": int, "tokens": int, "words": int, "lines": int}
        """
        return {
            "chars": len(text),
            "tokens": self.count_tokens(text),
            "words": len(text.split()),
            "lines": text.count("\n") + 1,
        }


# 全局单例 (延迟初始化)
_context_manager: ContextManager | None = None


def get_context_manager(max_context_tokens: int = 120_000) -> ContextManager:
    """获取全局 ContextManager 实例"""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager(
            max_context_tokens=max_context_tokens)
    return _context_manager
