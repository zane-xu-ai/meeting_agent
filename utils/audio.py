"""会议助手 - 音频格式转换工具"""

import subprocess
import wave
from pathlib import Path

from pydub import AudioSegment


def preprocess_audio(
    input_path: str | Path, sample_rate: int = 16000, output_path: Path | None = None,
) -> tuple[Path, float]:
    """Normalize on disk with bounded, local-only FFmpeg input."""
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    target = output_path or source.with_suffix(".preprocessed.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
             "-protocol_whitelist", "file,pipe", "-y", "-i", str(source),
             "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(target)],
            check=True, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        with wave.open(str(target)) as audio:
            return target, audio.getnframes() / audio.getframerate()
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def get_audio_duration_ms(audio_path: str | Path) -> int:
    """获取音频时长(毫秒)"""
    audio = AudioSegment.from_file(str(audio_path))
    return len(audio)
