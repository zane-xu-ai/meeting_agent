"""Run local ASR through the GPU worker without adding CUDA packages to Web."""

import os
import subprocess
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

from config import settings
from gpu_ops.runner import run_leased
from models.schemas import Transcript


LOCAL_MODEL = "local-whisper"
GPU_PROFILE = "whisper-small"
CURRENT_LEASE = ContextVar("gpu_lease", default=None)


def available():
    return bool(settings.amqp_url and settings.database_url and settings.gpu_database_url
                and settings.local_asr_model and settings.local_asr_python
                and settings.local_asr_model.is_dir() and settings.local_asr_python.is_file())


def model_fingerprint():
    path = settings.local_asr_model
    if not path:
        return None
    # The model directory is pinned by configuration. Include file metadata to detect replacement.
    return [str(path.resolve()), [(item.name, item.stat().st_size, item.stat().st_mtime_ns)
                                  for item in sorted(path.iterdir()) if item.is_file()]]


@lru_cache(maxsize=4)
def cuda_environment(python):
    libraries = subprocess.check_output([
        str(python), "-c",
        "import nvidia.cublas.lib, nvidia.cudnn.lib; "
        "print(nvidia.cublas.lib.__path__[0] + ':' + nvidia.cudnn.lib.__path__[0])",
    ], text=True, timeout=15).strip()
    previous = os.environ.get("LD_LIBRARY_PATH")
    return {"LD_LIBRARY_PATH": libraries + (":" + previous if previous else "")}


def transcribe(source):
    owned = CURRENT_LEASE.get()
    if not owned or not available():
        raise RuntimeError("本地 ASR 需要 GPU 队列和租约")
    scheduler, lease = owned
    script = Path(__file__).resolve().parents[1] / "gpu_ops" / "whisper_runner.py"
    with TemporaryDirectory() as temporary:
        result = Path(temporary) / "transcript.json"
        for beam in (3, 1):
            with TemporaryFile() as stderr:
                status = run_leased(scheduler, lease, [
                    str(settings.local_asr_python), str(script),
                    str(settings.local_asr_model), str(source), str(result), "cuda", str(beam),
                ], stderr=stderr, timeout=900, extra_env=cuda_environment(settings.local_asr_python))
                if status == 0:
                    return Transcript.model_validate_json(result.read_bytes())
                stderr.seek(0)
                error = stderr.read(65536).lower()
                if b"out of memory" not in error:
                    raise RuntimeError("本地 ASR 失败")
                if beam == 1:
                    raise MemoryError("本地 ASR 显存不足")
