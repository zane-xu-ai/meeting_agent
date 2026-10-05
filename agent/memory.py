"""
M4: Short-term Memory — 会话级记忆管理

负责:
- 每个 task_id 独立的对话历史管理
- 滑动窗口: 基于 token 预算保留最近对话
- 历史压缩: token 数超过阈值时，将早期对话压缩为摘要
- 记忆持久化: 内存存储 (可扩展到 Redis/DB)

设计原则:
- 所有窗口/阈值基于 token 数而非轮数，因为每轮对话长度差异巨大
  (3 轮长对话可能 = 30k tokens，30 轮短对话可能 = 3k tokens)
- 与 ContextManager (M3) 集成，复用 tiktoken 精确计算
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from loguru import logger

from agent.context import get_context_manager


@dataclass
class ConversationMemory:
    """
    单个任务的对话记忆

    结构:
    - summary: 早期对话的压缩摘要 (可选)
    - messages: 最近的对话历史 (滑动窗口)

    压缩策略:
    - 基于 token 数而非轮数，因为每轮对话长度差异巨大
    - 当 messages 的总 token 数超过 compress_threshold_tokens 时触发压缩
    - 压缩后保留最近 keep_token_budget 范围内的消息
    """
    task_id: str
    summary: str = ""
    messages: list[dict] = field(default_factory=list)
    compress_threshold_tokens: int = 50_000  # 超过 50k tokens 时触发压缩
    keep_token_budget: int = 20_000  # 压缩后保留最近 20k tokens 的消息

    def add_message(self, role: str, content: str) -> None:
        """
        添加一条消息到对话历史

        Args:
            role: "user" 或 "assistant"
            content: 消息内容
        """
        self.messages.append({"role": role, "content": content})
        logger.debug(f"[记忆 {self.task_id}] 添加消息: {role} ({len(content)} 字符)")

    def _count_tokens(self, text: str) -> int:
        """使用 ContextManager 计算 token 数"""
        ctx_mgr = get_context_manager()
        return ctx_mgr.count_tokens(text)

    def get_token_count(self) -> int:
        """
        计算当前 messages 的总 token 数

        Returns:
            所有消息的 token 总数
        """
        total = 0
        for msg in self.messages:
            total += self._count_tokens(msg.get("content", ""))
        return total

    def get_recent_messages_by_tokens(
        self, token_budget: int = 20_000
    ) -> list[dict]:
        """
        基于 token 预算获取最近的消息 (从尾部向前取，直到超出预算)

        这比固定轮数更精确：短对话保留更多轮，长对话保留更少轮。

        Args:
            token_budget: token 预算

        Returns:
            预算内的最近消息列表
        """
        if not self.messages:
            return []

        result: list[dict] = []
        total_tokens = 0

        # 从尾部向前遍历
        for msg in reversed(self.messages):
            msg_tokens = self._count_tokens(msg.get("content", ""))
            if total_tokens + msg_tokens > token_budget and result:
                break  # 超出预算，停止
            result.append(msg)
            total_tokens += msg_tokens

        result.reverse()  # 恢复时间顺序
        logger.debug(
            f"[记忆 {self.task_id}] token 窗口: "
            f"{len(result)} 条消息, {total_tokens} tokens "
            f"(预算 {token_budget})"
        )
        return result

    def get_recent_messages(self, n_rounds: int = 10) -> list[dict]:
        """
        获取最近 N 轮对话 (每轮 = user + assistant)

        注意: 这是简化接口，推荐用 get_recent_messages_by_tokens()

        Args:
            n_rounds: 轮数 (默认 10 轮 = 20 条消息)

        Returns:
            最近的消息列表
        """
        max_messages = n_rounds * 2
        return self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages.copy()

    def get_all_messages(self) -> list[dict]:
        """
        获取所有消息 (包括摘要)

        Returns:
            如果有摘要，返回 [summary_message] + messages
            否则返回 messages 的拷贝
        """
        if self.summary:
            summary_msg = {
                "role": "system",
                "content": f"[早期对话摘要] {self.summary}"
            }
            return [summary_msg] + self.messages
        return self.messages.copy()

    def get_message_count(self) -> int:
        """获取消息总数"""
        return len(self.messages)

    def get_round_count(self) -> int:
        """获取对话轮数 (每轮 = user + assistant)"""
        return len(self.messages) // 2

    def needs_compression(self) -> bool:
        """
        检查是否需要压缩 (基于 token 数)

        Returns:
            True 如果 messages 的总 token 数超过压缩阈值
        """
        token_count = self.get_token_count()
        if token_count > self.compress_threshold_tokens:
            logger.info(
                f"[记忆 {self.task_id}] 需要压缩: "
                f"{token_count} tokens > {self.compress_threshold_tokens} 阈值"
            )
            return True
        return False

    def compress_early_messages(self) -> list[dict]:
        """
        提取需要压缩的早期消息 (基于 token 预算)

        从头部开始取消息，直到剩余消息的 token 数 <= keep_token_budget

        Returns:
            需要压缩的早期消息列表
        """
        if not self.messages:
            return []

        # 计算总 token 数
        total_tokens = self.get_token_count()
        if total_tokens <= self.keep_token_budget:
            return []  # 不需要压缩

        # 从头部开始，找出需要压缩的消息
        early_messages: list[dict] = []
        remaining_tokens = total_tokens

        for msg in self.messages:
            msg_tokens = self._count_tokens(msg.get("content", ""))
            # 如果去掉这条消息后，剩余 token 数在预算内，且还有更多消息
            if remaining_tokens - msg_tokens <= self.keep_token_budget:
                # 这条消息可能是最后一条需要压缩的，也可能不需要
                # 检查去掉它之后剩余的消息是否足够
                break
            early_messages.append(msg)
            remaining_tokens -= msg_tokens

        return early_messages

    def apply_compression(self, summary: str) -> None:
        """
        应用压缩结果: 移除已压缩的早期消息，保留摘要

        Args:
            summary: 早期对话的摘要
        """
        early_messages = self.compress_early_messages()
        n_removed = len(early_messages)

        if n_removed == 0:
            logger.debug(f"[记忆 {self.task_id}] 无需压缩，消息已在预算内")
            return

        # 保存摘要
        if self.summary:
            self.summary = f"{self.summary}\n{summary}"
        else:
            self.summary = summary

        # 移除已压缩的消息
        self.messages = self.messages[n_removed:]

        remaining_tokens = self.get_token_count()
        logger.info(
            f"[记忆 {self.task_id}] 压缩完成: "
            f"移除 {n_removed} 条消息, "
            f"保留 {len(self.messages)} 条 ({remaining_tokens} tokens), "
            f"摘要 {len(summary)} 字符"
        )

    def clear(self) -> None:
        """清空记忆"""
        self.summary = ""
        self.messages = []
        logger.info(f"[记忆 {self.task_id}] 记忆已清空")

    def to_dict(self) -> dict:
        """
        序列化为字典

        Returns:
            {"task_id": str, "summary": str, "messages": list,
             "rounds": int, "tokens": int}
        """
        return {
            "task_id": self.task_id,
            "summary": self.summary,
            "messages": self.messages,
            "rounds": self.get_round_count(),
            "tokens": self.get_token_count(),
        }


class MemoryManager:
    """
    全局记忆管理器

    管理所有任务的对话记忆，提供线程安全的访问。
    """

    def __init__(
        self,
        compress_threshold_tokens: int = 50_000,
        keep_token_budget: int = 20_000,
    ):
        """
        Args:
            compress_threshold_tokens: 触发压缩的 token 阈值 (默认 50k)
                当 messages 总 token 数超过此值时触发压缩
            keep_token_budget: 压缩后保留的 token 预算 (默认 20k)
                压缩后保留最近的 N 条消息，直到 token 数 <= 此值
        """
        self._memories: dict[str, ConversationMemory] = {}
        self._lock = threading.Lock()
        self._compress_threshold_tokens = compress_threshold_tokens
        self._keep_token_budget = keep_token_budget
        logger.info(
            f"MemoryManager 初始化: "
            f"compress_threshold={compress_threshold_tokens} tokens, "
            f"keep_budget={keep_token_budget} tokens"
        )

    def get_memory(self, task_id: str) -> ConversationMemory:
        """
        获取指定任务的记忆 (不存在则创建)

        Args:
            task_id: 任务 ID

        Returns:
            ConversationMemory 实例
        """
        with self._lock:
            if task_id not in self._memories:
                self._memories[task_id] = ConversationMemory(
                    task_id=task_id,
                    compress_threshold_tokens=self._compress_threshold_tokens,
                    keep_token_budget=self._keep_token_budget,
                )
                logger.info(f"[记忆] 创建新记忆: task_id={task_id}")
            return self._memories[task_id]

    def add_message(self, task_id: str, role: str, content: str) -> None:
        """
        添加消息到指定任务的记忆

        Args:
            task_id: 任务 ID
            role: "user" 或 "assistant"
            content: 消息内容
        """
        memory = self.get_memory(task_id)
        memory.add_message(role, content)

    def get_recent_messages(
        self, task_id: str, token_budget: int = 20_000
    ) -> list[dict]:
        """
        获取指定任务的最近对话 (基于 token 预算)

        Args:
            task_id: 任务 ID
            token_budget: token 预算 (默认 20k)

        Returns:
            预算内的最近消息列表
        """
        memory = self.get_memory(task_id)
        return memory.get_recent_messages_by_tokens(token_budget)

    def get_all_messages(self, task_id: str) -> list[dict]:
        """
        获取指定任务的所有消息 (包括摘要)

        Args:
            task_id: 任务 ID

        Returns:
            消息列表
        """
        memory = self.get_memory(task_id)
        return memory.get_all_messages()

    def needs_compression(self, task_id: str) -> bool:
        """
        检查指定任务是否需要压缩

        Args:
            task_id: 任务 ID

        Returns:
            True 如果需要压缩
        """
        memory = self.get_memory(task_id)
        return memory.needs_compression()

    def clear_memory(self, task_id: str) -> None:
        """
        清空指定任务的记忆

        Args:
            task_id: 任务 ID
        """
        with self._lock:
            if task_id in self._memories:
                self._memories[task_id].clear()
                logger.info(f"[记忆] 清空记忆: task_id={task_id}")

    def get_stats(self, task_id: str) -> dict:
        """
        获取指定任务的记忆统计

        Args:
            task_id: 任务 ID

        Returns:
            {"rounds": int, "messages": int, "has_summary": bool}
        """
        memory = self.get_memory(task_id)
        return {
            "rounds": memory.get_round_count(),
            "messages": memory.get_message_count(),
            "tokens": memory.get_token_count(),
            "has_summary": bool(memory.summary),
            "summary_length": len(memory.summary) if memory.summary else 0,
        }

    def list_tasks(self) -> list[str]:
        """
        列出所有有记忆的任务 ID

        Returns:
            任务 ID 列表
        """
        with self._lock:
            return list(self._memories.keys())


# 全局单例
_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """获取全局 MemoryManager 实例"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(
            compress_threshold_tokens=50_000,
            keep_token_budget=20_000,
        )
    return _memory_manager
