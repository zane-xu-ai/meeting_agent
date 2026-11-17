"""Single leased request against host Ollama. Read the user prompt from stdin."""

import argparse
import sys
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--context", type=int, choices=(512, 1024, 2048, 4096), default=512)
    args = parser.parse_args()
    with httpx.Client(base_url="http://127.0.0.1:11434", trust_env=False, timeout=180) as client:
        resident = client.get("/api/ps").raise_for_status().json()["models"]
        if resident:
            raise SystemExit("当前 Ollama 有驻留模型，请等待现有请求释放")
        reply = client.post("/api/generate", json={
            "model": args.model, "prompt": sys.stdin.read(), "stream": False,
            "keep_alive": 0, "options": {"num_ctx": args.context, "num_predict": 256},
        }).raise_for_status().json()
        deadline = time.monotonic() + 15
        while client.get("/api/ps").raise_for_status().json()["models"]:
            if time.monotonic() >= deadline:
                raise SystemExit("Ollama 模型仍驻留，调度器将按外部显存占用处理")
            time.sleep(.5)
        print(reply["response"])


if __name__ == "__main__":
    main()
