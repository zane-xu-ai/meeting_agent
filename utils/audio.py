"""会议助手 - 音频格式转换工具"""

from pathlib import Path

from loguru import logger
from pydub import AudioSegment


def preprocess_audio(input_path: str | Path, sample_rate: int = 16000) -> Path:
    """
    音频预处理: 转换为单声道、指定采样率的 WAV 文件。

    Args:
        input_path: 输入音频文件路径
        sample_rate: 目标采样率

    Returns:
        预处理后的 WAV 文件路径
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {input_path}")

    output_path = input_path.with_suffix(".preprocessed.wav")

    logger.info(f"预处理音频: {input_path.name} -> 单声道 {sample_rate}Hz WAV")

    audio = AudioSegment.from_file(str(input_path))
    audio = audio.set_channels(1)  # 单声道(说话人分离要求)
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_sample_width(2)  # 16-bit
    audio.export(str(output_path), format="wav")

    duration_sec = len(audio) / 1000
    logger.info(f"预处理完成: 时长 {duration_sec:.1f}s, 输出 {output_path.name}")

    return output_path


def get_audio_duration_ms(audio_path: str | Path) -> int:
    """获取音频时长(毫秒)"""
    audio = AudioSegment.from_file(str(audio_path))
    return len(audio)
