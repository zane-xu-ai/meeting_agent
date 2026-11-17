"""Run in the pre-existing faster-whisper environment. Output JSON only."""

import json
import sys

from faster_whisper import WhisperModel


def main():
    model_path, source, output, device, beam = sys.argv[1:]
    model = WhisperModel(
        model_path, device=device, compute_type="int8_float16" if device == "cuda" else "int8",
        cpu_threads=2, local_files_only=True,
    )
    chunks, metadata = model.transcribe(source, beam_size=int(beam), vad_filter=True)
    result = {
        "duration_ms": int(metadata.duration * 1000),
        "sentences": [{"text": chunk.text.strip(), "begin_time": int(chunk.start * 1000),
                       "end_time": int(chunk.end * 1000)} for chunk in chunks],
    }
    with open(output, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False)


if __name__ == "__main__":
    main()
