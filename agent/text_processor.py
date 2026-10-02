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
        self, transcript: Transcript, window_minutes: int = 5
    ) -> list[str]:
        """
        按时间窗口将转写文本切片，用于长文本分段摘要。

        Args:
            transcript: 清洗后的转写结果
            window_minutes: 每个切片的时间窗口(分钟)

        Returns:
            文本切片列表
        """
        window_ms = window_minutes * 60 * 1000
        chunks: list[str] = []
        current_chunk: list[str] = []
        chunk_start = 0

        for s in transcript.sentences:
            if s.begin_time - chunk_start >= window_ms and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                chunk_start = s.begin_time

            speaker = s.speaker or "未知"
            current_chunk.append(f"[{speaker}] {s.text}")

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        logger.info(f"文本分片: {len(chunks)} 个切片 (每 {window_minutes} 分钟)")
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
