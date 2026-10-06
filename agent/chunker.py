"""
M5: 父子块分块器 — Summary 和 ASR 的父子块切分

设计原则:
- 子块 (Child): 用于 Embedding + 向量检索 (精确匹配)
- 父块 (Parent): 检索命中后，将父块注入 LLM 上下文 (提供完整语境)

Summary 分块规则:
- 每个 H2 段落为一个父块
- H2 有 H3 时: 每个子块 = H2 标题 + 单个 H3 内容
- H2 无 H3 时: 子块 = 父块 (父子相同)

ASR 分块规则:
- 子块 = ASR 单行 (含时间戳 + 说话人)
- 父块 = 连续子块累积到 ~500 字符
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger


@dataclass
class ChildChunk:
    """子块 — 用于 Embedding 和向量检索"""
    child_id: str           # 子块唯一 ID
    parent_id: str          # 所属父块 ID
    content: str            # 子块内容 (用于 Embedding)
    chunk_index: int        # 在父块内的序号


@dataclass
class ParentChunk:
    """父块 — 用于提供完整上下文"""
    parent_id: str          # 父块唯一 ID
    content: str            # 父块完整内容
    task_id: str            # 所属任务 ID
    rec_type: str           # meeting / interview / general
    source_type: str        # "summary" | "asr"
    source_file: str        # 原始文件名
    created_at: str         # 创建时间 (ISO 格式)
    section: str            # 所属章节 (如 "全文摘要", "章节速览")
    metadata: dict = field(default_factory=dict)  # 额外元数据
    children: list[ChildChunk] = field(default_factory=list)


def _gen_id() -> str:
    """生成唯一 ID"""
    return str(uuid.uuid4())[:12]


def _now_iso() -> str:
    """当前时间 ISO 格式"""
    return datetime.now().isoformat(timespec="seconds")


# ═══════════════════════════════════════════════════════════════
# Summary 父子块分块
# ═══════════════════════════════════════════════════════════════

def chunk_summary(
    text: str,
    task_id: str,
    rec_type: str,
    source_file: str,
) -> list[ParentChunk]:
    """
    Summary 父子块分块

    规则:
    - 每个 H2 段落为一个父块
    - H2 有 H3 时: 每个子块 = H2 标题 + 单个 H3 内容
    - H2 无 H3 时: 子块 = 父块 (父子相同)

    Args:
        text: Markdown 格式的分析报告
        task_id: 任务 ID
        rec_type: 录音类型 (meeting/interview/general)
        source_file: 原始文件名

    Returns:
        父块列表 (每个父块包含子块列表)
    """
    logger.debug(
        f"[Chunker-Summary] 开始分块: task_id={task_id}, "
        f"rec_type={rec_type}, 文本长度={len(text)}"
    )

    # 按 H2 标题切分段落
    h2_sections = re.split(r'\n(?=## )', text)
    h2_sections = [s.strip() for s in h2_sections if s.strip()]

    parent_chunks: list[ParentChunk] = []
    total_children = 0

    for section_idx, section in enumerate(h2_sections):
        lines = section.split('\n')
        h2_heading = lines[0].strip().lstrip('#').strip()

        # 跳过 H1 标题 (通常是 "会议纪要 - 2026-10-03 10:44")
        if lines[0].startswith('# ') and not lines[0].startswith('## '):
            logger.debug(f"[Chunker-Summary] 跳过 H1: {h2_heading[:30]}")
            continue

        # 检查是否有 H3 子段落
        h3_parts = re.split(r'\n(?=### )', section)
        h3_parts = [p.strip() for p in h3_parts if p.strip()]
        has_h3 = len(h3_parts) > 1  # 第一个是 H2 本身

        parent_id = _gen_id()
        parent_content = section
        created_at = _now_iso()

        parent = ParentChunk(
            parent_id=parent_id,
            content=parent_content,
            task_id=task_id,
            rec_type=rec_type,
            source_type="summary",
            source_file=source_file,
            created_at=created_at,
            section=h2_heading,
            metadata={"section_index": section_idx},
        )

        if has_h3:
            # 有 H3: 每个子块 = H2 标题 + 单个 H3
            for h3_idx, h3_part in enumerate(h3_parts[1:], 1):
                h3_lines = h3_part.split('\n')
                h3_heading = h3_lines[0].strip().lstrip('#').strip()

                # 子块内容: H2 标题 + H3 标题 + H3 内容
                child_content = f"{h2_heading} - {h3_part}"

                child = ChildChunk(
                    child_id=_gen_id(),
                    parent_id=parent_id,
                    content=child_content,
                    chunk_index=h3_idx,
                )
                parent.children.append(child)

            logger.debug(
                f"[Chunker-Summary] 父块 [{h2_heading}]: "
                f"{len(parent.children)} 个子块"
            )
        else:
            # 无 H3: 子块 = 父块 (父子相同)
            child = ChildChunk(
                child_id=_gen_id(),
                parent_id=parent_id,
                content=parent_content,
                chunk_index=0,
            )
            parent.children.append(child)

            logger.debug(
                f"[Chunker-Summary] 父块 [{h2_heading}]: "
                f"无 H3, 父子相同"
            )

        total_children += len(parent.children)
        parent_chunks.append(parent)

    logger.info(
        f"[Chunker-Summary] 完成: {len(parent_chunks)} 个父块, "
        f"{total_children} 个子块"
    )
    return parent_chunks


# ═══════════════════════════════════════════════════════════════
# ASR 父子块分块
# ═══════════════════════════════════════════════════════════════

def chunk_asr(
    text: str,
    task_id: str,
    rec_type: str,
    source_file: str,
    parent_target_chars: int = 500,
) -> list[ParentChunk]:
    """
    ASR 父子块分块

    规则:
    - 子块 = ASR 单行 (含时间戳 + 说话人)
    - 父块 = 连续子块累积到 ~500 字符

    Args:
        text: ASR 转写文本 (每行格式: [00:00 -> 00:08] Speaker-00: ...)
        task_id: 任务 ID
        rec_type: 录音类型
        source_file: 原始文件名
        parent_target_chars: 父块目标字符数 (默认 500)

    Returns:
        父块列表
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    logger.debug(
        f"[Chunker-ASR] 开始分块: task_id={task_id}, "
        f"总行数={len(lines)}, 目标父块大小={parent_target_chars} 字符"
    )

    if not lines:
        logger.warning("[Chunker-ASR] 文本为空")
        return []

    parent_chunks: list[ParentChunk] = []
    parent_id = _gen_id()
    current_children: list[ChildChunk] = []
    current_chars = 0
    parent_idx = 0

    for line_idx, line in enumerate(lines):
        # 子块 = 单行
        child = ChildChunk(
            child_id=_gen_id(),
            parent_id=parent_id,
            content=line,
            chunk_index=len(current_children),
        )
        current_children.append(child)
        current_chars += len(line)

        # 达到目标大小，切分父块
        if current_chars >= parent_target_chars:
            parent = ParentChunk(
                parent_id=parent_id,
                content='\n'.join(c.content for c in current_children),
                task_id=task_id,
                rec_type=rec_type,
                source_type="asr",
                source_file=source_file,
                created_at=_now_iso(),
                section=f"ASR 片段 {parent_idx + 1}",
                metadata={
                    "parent_index": parent_idx,
                    "line_start": line_idx - len(current_children) + 2,
                    "line_end": line_idx + 1,
                },
                children=current_children,
            )
            parent_chunks.append(parent)

            logger.debug(
                f"[Chunker-ASR] 父块 {parent_idx + 1}: "
                f"行 {parent.metadata['line_start']}-{parent.metadata['line_end']}, "
                f"{len(current_children)} 子块, {current_chars} 字符"
            )

            # 重置
            parent_idx += 1
            parent_id = _gen_id()
            current_children = []
            current_chars = 0

    # 处理剩余子块
    if current_children:
        parent = ParentChunk(
            parent_id=parent_id,
            content='\n'.join(c.content for c in current_children),
            task_id=task_id,
            rec_type=rec_type,
            source_type="asr",
            source_file=source_file,
            created_at=_now_iso(),
            section=f"ASR 片段 {parent_idx + 1}",
            metadata={
                "parent_index": parent_idx,
                "line_start": len(lines) - len(current_children) + 1,
                "line_end": len(lines),
            },
            children=current_children,
        )
        parent_chunks.append(parent)

        logger.debug(
            f"[Chunker-ASR] 父块 {parent_idx + 1} (末尾): "
            f"行 {parent.metadata['line_start']}-{parent.metadata['line_end']}, "
            f"{len(current_children)} 子块, {current_chars} 字符"
        )

    total_children = sum(len(p.children) for p in parent_chunks)
    logger.info(
        f"[Chunker-ASR] 完成: {len(parent_chunks)} 个父块, "
        f"{total_children} 个子块"
    )
    return parent_chunks
