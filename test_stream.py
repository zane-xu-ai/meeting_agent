#!/usr/bin/env python3
"""测试流式输出"""
import requests
import time

url = "http://localhost:9090/api/stream/ae680f8f26d3"
start_time = time.time()

with requests.get(url, stream=True) as r:
    r.raise_for_status()
    for line in r.iter_lines():
        if line:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.3f}s] {line.decode('utf-8')[:100]}")
