"""
M4: Short-term Memory — 会话级记忆管理 (三层分级压缩)

压缩策略:
  ┌───────────────────────────────────────────────────────────┐
  │ L1: 工具输出即时蒸馏 (Tool Observation Pruning)            │
  │     单条消息 > prune_threshold → 即刻截断/蒸馏              │
  ├───────────────────────────────────────────────────────────┤
  │ L2: 高低水位线滑动压缩 (High / Low Watermark)              │
  │     总量 > high_watermark → 压缩到 low_watermark           │
  │     类似 GC 的滞后机制，避免频繁压缩                        │
  ├───────────────────────────────────────────────────────────┤
  │ L3: 子任务里程碑归档 (Task Milestones)                     │
  │     语义阶段性触发：完成一个子任务后归档当前对话为摘要        │
  └───────────────────────────────────────────────────────────┘

设计原则:
  - 所有阈值基于 token 数 (tiktoken)
  - 压缩目标: 3000-5000 tokens
  - 与 M3 ContextManager 集成，复用精确 token 计算
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from loguru import logger

from agent.context import get_context_manager


# ═══════════════════════════════════════════════════════════════
# 常量定义
# ═══════════════════════════════════════════════════════════════

# L1: 单条消息蒸馏阈值 (超过此值即刻截断)
PRUNE_THRESHOLD = 2_000  # tokens

# L2: 高低水位线
HIGH_WATERMARK = 8_000   # tokens — 触发压缩
LOW_WATERMARK = 4_000    # tokens — 压缩目标

# L3: 里程碑归档保留量
MILESTONE_KEEP_TOKENS = 2_000  # tokens — 归档后保留最近的消息


# ═══════════════════════════════════════════════════════════════
# L1: 工具输出即时蒸馏
# ═══════════════════════════════════════════════════════════════

def distill_message(content: str, max_tokens: int = PRUNE_THRESHOLD) -> str:
    """
    对单条过长消息进行即时蒸馏

    当工具输出 (如 LLM 回复) 超过阈值时，立即截断并添加标记。
    后续可升级为 LLM 摘要蒸馏。

    Args:
        content: 原始消息内容
        max_tokens: 最大 token 数

    Returns:
        蒸馏后的消息内容
    """
    ctx = get_context_manager()
    tokens = ctx.count_tokens(content)

    if tokens <= max_tokens:
        return content

    # 截断到 max_tokens，保留首尾 (头部保留 70%，尾部保留 30%)
    head_budget = int(max_tokens * 0.7)
    tail_budget = int(max_tokens * 0.3)

    if ctx._encoding:
        encoded = ctx._encoding.encode(content)
        head = ctx._encoding.decode(encoded[:head_budget])
        tail = ctx._encoding.decode(encoded[-tail_budget:])
    else:
        # 降级: 按字符比例
        ratio = max_tokens / tokens
        char_limit = int(len(content) * ratio)
        head_chars = int(char_limit * 0.7)
        tail_chars = int(char_limit * 0.3)
        head = content[:head_chars]
        tail = content[-tail_chars:]

    distilled = (
        f"{head}\n\n"
        f"[... 已蒸馏 {tokens - max_tokens} tokens ...]\n\n"
        f"{tail}"
    )
    new_tokens = ctx.count_tokens(distilled)
    logger.info(
        f"[L1 蒸馏] {tokens} → {new_tokens} tokens "
        f"({len(content)} → {len(distilled)} 字符)"
    )
    return distilled


# ═══════════════════════════════════════════════════════════════
# ConversationMemory — 单个任务的对话记忆
# ═══════════════════════════════════════════════════════════════

@dataclass
class Milestone:
    """里程碑记录"""
    label: str
    summary: str
    message_index: int  # 归档时的消息索引
    token_count: int    # 归档时的 token 数


@dataclass
class ConversationMemory:
    """
    单个任务的对话记忆 (三层分级压缩)

    结构:
    - summary: 累积的全局摘要 (L2 压缩 + L3 归档)
    - messages: 当前活跃消息列表
    - milestones: 里程碑记录 (L3)
    """
    task_id: str
    summary: str = ""
    messages: list[dict] = field(default_factory=list)
    milestones: list[Milestone] = field(default_factory=list)

    # L1 配置
    prune_threshold: int = PRUNE_THRESHOLD

    # L2 配置
    high_watermark: int = HIGH_WATERMARK
    low_watermark: int = LOW_WATERMARK

    # L3 配置
    milestone_keep_tokens: int = MILESTONE_KEEP_TOKENS

    # ── Token 计算 ──────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """使用 ContextManager 计算 token 数"""
        return get_context_manager().count_tokens(text)

    def get_token_count(self) -> int:
        """计算当前 messages 的总 token 数"""
        return sum(
            self._count_tokens(msg.get("content", ""))
            for msg in self.messages
        )

    def get_message_count(self) -> int:
        """获取消息总数"""
        return len(self.messages)

    def get_round_count(self) -> int:
        """获取对话轮数"""
        return len(self.messages) // 2

    # ── 消息添加 (自动触发 L1) ──────────────────────────────

    def add_message(self, role: str, content: str) -> dict:
        """
        添加消息 (自动触发 L1 蒸馏)

        Args:
            role: "user" / "assistant" / "tool"
            content: 消息内容

        Returns:
            实际存储的消息 (可能被 L1 蒸馏)
        """
        # L1: 对非用户消息进行即时蒸馏
        # (用户消息不蒸馏，保持原始输入)
        if role != "user" and self._count_tokens(content) > self.prune_threshold:
            content = distill_message(content, self.prune_threshold)

        msg = {"role": role, "content": content}
        self.messages.append(msg)

        logger.debug(
            f"[记忆 {self.task_id}] 添加 {role}: "
            f"{self._count_tokens(content)} tokens"
        )
        return msg

    # ── 获取消息 ────────────────────────────────────────────

    def get_recent_messages_by_tokens(
        self, token_budget: int = LOW_WATERMARK
    ) -> list[dict]:
        """
        基于 token 预算获取最近的消息

        从尾部向前取，直到超出预算。

        Args:
            token_budget: token 预算

        Returns:
            预算内的最近消息列表
        """
        if not self.messages:
            return []

        result: list[dict] = []
        total = 0

        for msg in reversed(self.messages):
            t = self._count_tokens(msg.get("content", ""))
            if total + t > token_budget and result:
                break
            result.append(msg)
            total += t

        result.reverse()
        return result

    def get_all_messages(self) -> list[dict]:
        """获取所有消息 (包括全局摘要)"""
        if self.summary:
            return [
                {"role": "system", "content": f"[对话摘要] {self.summary}"}
            ] + self.messages
        return self.messages.copy()

    # ── L2: 高低水位线滑动压缩 ──────────────────────────────

    def needs_compression(self) -> bool:
        """
        L2: 检查是否需要压缩 (高水位线触发)

        Returns:
            True 如果总 token 数超过 high_watermark
        """
        tokens = self.get_token_count()
        if tokens > self.high_watermark:
            logger.info(
                f"[L2 水位线] {tokens} tokens > "
                f"high_watermark {self.high_watermark}"
            )
            return True
        return False

    def get_messages_to_compress(self) -> list[dict]:
        """
        L2: 获取需要压缩的早期消息

        保留 low_watermark 以内的最近消息，其余返回用于压缩。

        Returns:
            需要压缩的早期消息列表
        """
        if not self.messages:
            return []

        # 从尾部向前，找到 low_watermark 边界
        keep: list[dict] = []
        keep_tokens = 0

        for msg in reversed(self.messages):
            t = self._count_tokens(msg.get("content", ""))
            if keep_tokens + t > self.low_watermark and keep:
                break
            keep.append(msg)
            keep_tokens += t

        n_keep = len(keep)
        if n_keep >= len(self.messages):
            return []  # 全部保留，无需压缩

        return self.messages[:-n_keep]

    def apply_compression(self, summary: str) -> None:
        """
        L2: 应用压缩结果

        Args:
            summary: 早期消息的摘要
        """
        to_compress = self.get_messages_to_compress()
        if not to_compress:
            return

        n_removed = len(to_compress)

        # 累积摘要
        if self.summary:
            self.summary = f"{self.summary}\n{summary}"
        else:
            self.summary = summary

        # 移除已压缩的消息
        self.messages = self.messages[n_removed:]

        remaining = self.get_token_count()
        logger.info(
            f"[L2 压缩] 移除 {n_removed} 条消息, "
            f"保留 {len(self.messages)} 条 ({remaining} tokens), "
            f"摘要 +{len(summary)} 字符"
        )

    # ── L3: 里程碑归档 ──────────────────────────────────────

    def mark_milestone(self, label: str, summary: str = "") -> Milestone:
        """
        L3: 标记里程碑并归档当前对话

        当一个子任务完成时调用，将当前对话压缩为摘要，
        只保留最近的少量消息作为活跃上下文。

        Args:
            label: 里程碑标签 (如 "分析完成", "待办提取完成")
            summary: 归档摘要 (为空则自动生成)

        Returns:
            Milestone 记录
        """
        current_tokens = self.get_token_count()

        # 生成归档摘要
        if not summary:
            # 简单截取前几条消息作为摘要
            early = self.messages[:-4] if len(self.messages) > 4 else []
            if early:
                texts = [m["content"][:200] for m in early[:6]]
                summary = f"[{label}] " + " | ".join(texts)
            else:
                summary = f"[{label}] 对话归档"

        # 保留最近的消息
        keep: list[dict] = []
        keep_tokens = 0
        for msg in reversed(self.messages):
            t = self._count_tokens(msg.get("content", ""))
            if keep_tokens + t > self.milestone_keep_tokens and keep:
                break
            keep.append(msg)
            keep_tokens += t

        n_archived = len(self.messages) - len(keep)

        # 记录里程碑
        milestone = Milestone(
            label=label,
            summary=summary,
            message_index=n_archived,
            token_count=current_tokens,
        )
        self.milestones.append(milestone)

        # 累积到全局摘要
        if self.summary:
            self.summary = f"{self.summary}\n{summary}"
        else:
            self.summary = summary

        # 截断消息
        self.messages = self.messages[-len(keep):] if keep else []

        logger.info(
            f"[L3 里程碑] '{label}': "
            f"归档 {n_archived} 条消息, "
            f"保留 {len(self.messages)} 条 ({keep_tokens} tokens)"
        )
        return milestone

    # ── 通用方法 ────────────────────────────────────────────

    def clear(self) -> None:
        """清空记忆"""
        self.summary = ""
        self.messages = []
        self.milestones = []
        logger.info(f"[记忆 {self.task_id}] 记忆已清空")

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "task_id": self.task_id,
            "summary": self.summary,
            "messages": self.messages,
            "rounds": self.get_round_count(),
            "tokens": self.get_token_count(),
            "milestones": [
                {"label": m.label, "token_count": m.token_count}
                for m in self.milestones
            ],
        }


# ═══════════════════════════════════════════════════════════════
# MemoryManager — 全局记忆管理器
# ═══════════════════════════════════════════════════════════════

class MemoryManager:
    """
    全局记忆管理器

    管理所有任务的对话记忆，提供线程安全的访问。
    集成三层压缩策略。
    """

    def __init__(
        self,
        prune_threshold: int = PRUNE_THRESHOLD,
        high_watermark: int = HIGH_WATERMARK,
        low_watermark: int = LOW_WATERMARK,
    ):
        self._memories: dict[str, ConversationMemory] = {}
        self._lock = threading.Lock()
        self._prune_threshold = prune_threshold
        self._high_watermark = high_watermark
        self._low_watermark = low_watermark
        logger.info(
            f"MemoryManager 初始化: "
            f"L1(prune={prune_threshold}), "
            f"L2(high={high_watermark}, low={low_watermark}), "
            f"L3(milestone_keep={MILESTONE_KEEP_TOKENS})"
        )

    def get_memory(self, task_id: str) -> ConversationMemory:
        """获取指定任务的记忆 (不存在则创建)"""
        with self._lock:
            if task_id not in self._memories:
                self._memories[task_id] = ConversationMemory(
                    task_id=task_id,
                    prune_threshold=self._prune_threshold,
                    high_watermark=self._high_watermark,
                    low_watermark=self._low_watermark,
                )
                logger.info(f"[记忆] 创建新记忆: task_id={task_id}")
            return self._memories[task_id]

    def add_message(self, task_id: str, role: str, content: str) -> dict:
        """添加消息 (自动触发 L1 蒸馏)"""
        memory = self.get_memory(task_id)
        return memory.add_message(role, content)

    def get_recent_messages(
        self, task_id: str, token_budget: int = LOW_WATERMARK
    ) -> list[dict]:
        """获取最近消息 (基于 token 预算)"""
        memory = self.get_memory(task_id)
        return memory.get_recent_messages_by_tokens(token_budget)

    # ── L2 接口 ─────────────────────────────────────────────

    def needs_compression(self, task_id: str) -> bool:
        """L2: 检查是否需要压缩"""
        return self.get_memory(task_id).needs_compression()

    def compress(self, task_id: str, summary: str) -> None:
        """L2: 应用压缩"""
        self.get_memory(task_id).apply_compression(summary)

    # ── L3 接口 ─────────────────────────────────────────────

    def mark_milestone(
        self, task_id: str, label: str, summary: str = ""
    ) -> Milestone:
        """L3: 标记里程碑"""
        return self.get_memory(task_id).mark_milestone(label, summary)

    # ── 通用接口 ────────────────────────────────────────────

    def clear_memory(self, task_id: str) -> None:
        """清空指定任务的记忆"""
        with self._lock:
            if task_id in self._memories:
                self._memories[task_id].clear()

    def get_stats(self, task_id: str) -> dict:
        """获取记忆统计"""
        memory = self.get_memory(task_id)
        return {
            "rounds": memory.get_round_count(),
            "messages": memory.get_message_count(),
            "tokens": memory.get_token_count(),
            "has_summary": bool(memory.summary),
            "summary_length": len(memory.summary) if memory.summary else 0,
            "milestones": len(memory.milestones),
            "high_watermark": memory.high_watermark,
            "low_watermark": memory.low_watermark,
        }

    def list_tasks(self) -> list[str]:
        """列出所有有记忆的任务 ID"""
        with self._lock:
            return list(self._memories.keys())


# ═══════════════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════════════

_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """获取全局 MemoryManager 实例"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
