"""会议助手 - 文本预处理模块"""

from __future__ import annotations

import re

from loguru import logger

from config import settings
from models.schemas import Sentence, Transcript


# 常见语气词/填充词模式
FILLER_WORDS = re.compile(
    r"^(嗯+|啊+|呃+|那个|就是说|然后那个|对对对|好好好|哦哦|嗯嗯|"
    r"这个|那什么|是吧|对吧|就是说呢|怎么说呢)[，,。、]?\s*"
)


class TextProcessor:
    """会议转写文本预处理"""

    def __init__(
        self,
        min_sentence_length: int | None = None,
    ):
        self.min_length = min_sentence_length or settings.min_sentence_length

    def clean(self, transcript: Transcript) -> Transcript:
        """
        文本清洗流水线:
        1. 去除语气词
        2. 合并短句
        3. 清理空白

        Args:
            transcript: ASR 原始转写结果

        Returns:
            清洗后的 Transcript
        """
        logger.info("开始文本预处理...")
        sentences = transcript.sentences

        # Step 1: 去除语气词
        sentences = [self._remove_fillers(s) for s in sentences]

        # Step 2: 过滤空句
        sentences = [s for s in sentences if s.text.strip()]

        # Step 3: 合并短句(连续同说话人的短句合并)
        sentences = self._merge_short_sentences(sentences)

        # Step 4: 清理文本
        for s in sentences:
            s.text = self._clean_text(s.text)

        result = Transcript(sentences=sentences,
                            duration_ms=transcript.duration_ms)
        logger.info(
            f"预处理完成: {len(transcript.sentences)} 句 -> {len(sentences)} 句")
        return result

    def chunk_by_time(
        self,
        transcript: Transcript,
        window_minutes: int = 5,
        overlap_sentences: int = 3,
    ) -> list[str]:
        """
        按时间窗口将转写文本切片，用于长文本分段摘要。

        采用滑动窗口设计：相邻切片之间保留 overlap_sentences 句重叠，
        确保切割边界处的上下文不会丢失。

        Args:
            transcript: 清洗后的转写结果
            window_minutes: 每个切片的时间窗口(分钟)
            overlap_sentences: 相邻切片之间的重叠句数

        Returns:
            文本切片列表
        """
        window_ms = window_minutes * 60 * 1000
        chunks: list[str] = []
        current_chunk: list[str] = []
        chunk_start = 0
        # 用于重叠的缓冲：保存每个句子的文本
        overlap_buffer: list[str] = []

        for s in transcript.sentences:
            if s.begin_time - chunk_start >= window_ms and current_chunk:
                chunks.append("\n".join(current_chunk))
                # 保留尾部 overlap 句作为下一片段的头部重叠
                overlap_buffer = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                current_chunk = []
                chunk_start = s.begin_time

                # 将重叠句添加到新片段的开头
                if overlap_buffer:
                    current_chunk.extend(overlap_buffer)

            speaker = s.speaker or "未知"
            current_chunk.append(f"[{speaker}] {s.text}")

        if current_chunk:
            # 去除纯重叠内容（与上一片段完全重复的尾部）
            chunks.append("\n".join(current_chunk))

        logger.info(
            f"文本分片: {len(chunks)} 个切片 "
            f"(每 {window_minutes} 分钟, 重叠 {overlap_sentences} 句)"
        )
        return chunks

    def chunk_by_sentences(
        self,
        text: str,
        chunk_size: int = 3000,
        overlap_sentences: int = 3,
    ) -> list[str]:
        """
        按句子边界将纯文本切片，采用滑动窗口重叠设计。

        与简单字符硬切不同，本方法：
        1. 先按换行符拆分为句子列表
        2. 贪心合并句子直到超过 chunk_size
        3. 相邻切片之间保留 overlap_sentences 句重叠

        Args:
            text: 待切片的纯文本 (句子以换行分隔)
            chunk_size: 每个切片的目标字符数
            overlap_sentences: 相邻切片之间的重叠句数

        Returns:
            文本切片列表
        """
        sentences = [line for line in text.split("\n") if line.strip()]
        if not sentences:
            return [text] if text.strip() else []

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent) + 1  # +1 for newline
            # 当前切片已满，且不是第一句 -> 切分
            if current_len + sent_len > chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                # 保留尾部重叠句
                overlap = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                current_chunk = list(overlap)
                current_len = sum(len(s) + 1 for s in overlap)

            current_chunk.append(sent)
            current_len += sent_len

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        logger.info(
            f"文本分片: {len(chunks)} 个切片 "
            f"(每 ~{chunk_size} 字, 重叠 {overlap_sentences} 句)"
        )
        return chunks

    def _remove_fillers(self, sentence: Sentence) -> Sentence:
        """去除句首语气词"""
        cleaned = FILLER_WORDS.sub("", sentence.text)
        if cleaned != sentence.text:
            sentence.text = cleaned
        return sentence

    def _merge_short_sentences(self, sentences: list[Sentence]) -> list[Sentence]:
        """合并连续同说话人的短句"""
        if not sentences:
            return sentences

        merged: list[Sentence] = []
        current = sentences[0].model_copy()

        for s in sentences[1:]:
            same_speaker = (
                current.speaker == s.speaker
                and len(current.text) < self.min_length
            )
            if same_speaker:
                current.text += s.text
                current.end_time = s.end_time
            else:
                merged.append(current)
                current = s.model_copy()

        merged.append(current)
        return merged

    @staticmethod
    def _clean_text(text: str) -> str:
        """清理文本: 去除多余空白、重复标点"""
        # 去除多余空白
        text = re.sub(r"\s+", " ", text).strip()
        # 去除重复标点
        text = re.sub(r"([，,。！？；;])\1+", r"\1", text)
        # 去除中文和英文之间的多余空格(可选)
        text = re.sub(r"\s+", " ", text)
        return text
