# LangGraph 流水线重构计划

## 目标

将 `server.py` 中 4 条重复处理路径（本地音频/视频、URL 音频/视频）统一为 **一个 LangGraph StateGraph**，同时保留 `pipeline.py` 中的格式化函数和分段分析策略。

## 图结构设计

```
START
  │
  ▼
[extract_audio]  ─── 视频→提取音频; 音频/URL→跳过
  │
  ▼
[preprocess]     ─── 本地文件→预处理; URL→跳过
  │
  ▼
[asr_transcribe] ─── 调用 ASR 转写
  │
  ▼
[text_clean]     ─── 文本预处理(去语气词/合并短句)
  │
  ▼
[classify]       ─── LLM 判断类型
  │
  ├── meeting    ──▶ [analyze_meeting]
  ├── interview  ──▶ [analyze_interview]
  └── general    ──▶ [analyze_general]
          │               │               │
          └───────┬───────┘───────────────┘
                  ▼
           [format_output] ─── 格式化 Markdown + 保存缓存
                  │
                  ▼
                 END
```

## State 设计

```python
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

class AgentState(BaseModel):
    # ── 输入 ──
    source: str                          # 文件路径 or URL
    source_type: str                     # "local_audio" | "local_video" | "url_audio" | "url_video"
    asr_model: str | None = None
    llm_model: str | None = None
    
    # ── 中间产物 ──
    audio_path: str | None = None        # 提取/预处理后的音频路径
    preprocessed_path: str | None = None # 预处理后的 WAV 路径
    asr_input: str | None = None         # ASR 输入(路径或URL)
    transcript_text: str = ""            # 转写文本(清洗后)
    asr_duration: float = 0.0
    rec_type: str = "general"            # meeting | interview | general
    
    # ── 分析结果 ──
    result_data: dict | None = None      # LLM 分析原始 JSON
    meeting_result: dict | None = None   # 会议结构化结果
    output_text: str = ""                # 最终 Markdown
    
    # ── 元信息 ──
    audio_stem: str = ""
    video_title: str | None = None
    llm_start: float = 0.0
    error: str | None = None
```

## 实施步骤

### Step 1: 安装依赖
- `requirements.txt` 添加 `langgraph>=0.4.0`
- `pip install langgraph`

### Step 2: 新建 `agent/graph.py` — LangGraph 图定义
- 定义 `AgentState` (TypedDict 或 Pydantic)
- 实现各节点函数（每个函数接收 State，返回部分 State 更新）：
  - `extract_audio(state)` — 视频提取音频，音频直接透传
  - `preprocess(state)` — 本地文件预处理为 16kHz WAV，URL 跳过
  - `asr_transcribe(state)` — 调用 ASRClient
  - `text_clean(state)` — 调用 TextProcessor.clean()
  - `classify(state)` — 调用 classify_recording()
  - `analyze_meeting(state)` / `analyze_interview(state)` / `analyze_general(state)` — 复用 pipeline.py 中的分析逻辑
  - `format_output(state)` — 格式化 Markdown + 保存缓存
- 定义条件路由 `route_by_type(state) -> "meeting" | "interview" | "general"`
- 构建 StateGraph 并编译

### Step 3: 重构 `server.py` — 用图替换 4 个处理函数
- 4 个 `_process_*` 函数合并为 1 个：根据输入判断 `source_type`，调用 `graph.ainvoke(state)`
- 保留 `_update()` 进度回调机制（在节点函数中调用）
- 保留前端轮询的 progress 值不变

### Step 4: 重构 `main.py` — CLI 也用图
- `transcript-only` 和 `analyze` 命令改为调用同一个图

### Step 5: 图可视化
- 利用 `graph.get_graph().draw_mermaid_png()` 或 `draw_ascii()` 生成流程图
- 保存到 `docs/graph.png`，README 中引用

### Step 6: 验证
- 确保 4 条路径（本地音频/视频、URL 音频/视频）功能不变
- 健康检查 + 实际测试

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| State 类型 | TypedDict | LangGraph 原生支持，比 Pydantic 更轻量 |
| 节点粒度 | 每个步骤一个节点 | 便于可视化、断点、后续插入 Tool 调用 |
| 异步 | 节点函数用 async | ASR/LLM 调用本身是异步的 |
| 进度回调 | 节点函数内调用 _update | 保持前端兼容，后续可改为 LangGraph callback |
| pipeline.py | 保留格式化函数，分析逻辑迁移到图节点 | 避免破坏现有导入 |

## 文件变更清单

| 文件 | 操作 |
|------|------|
| `requirements.txt` | 添加 langgraph |
| `agent/graph.py` | **新建** — StateGraph 定义 + 所有节点函数 |
| `server.py` | 重构 — 4 个处理函数合并为调用 graph |
| `main.py` | 重构 — CLI 命令改为调用 graph |
| `agent/pipeline.py` | 保留格式化函数，分析函数迁移到 graph.py 后标记废弃 |
| `agent/classifier.py` | 不变 — 被 graph 节点调用 |
| `agent/asr_client.py` | 不变 — 被 graph 节点调用 |
| `agent/llm_client.py` | 不变 — 被 graph 节点调用 |
