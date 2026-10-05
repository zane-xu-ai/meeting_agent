# Meeting Agent — Agent 化演进路线图

> LangGraph 流水线重构 (p01) 和流水线问题修复 (p02) 已完成。
> **M1 Function Calling 已完成** ✅ (2026-10-05)
> **M2 多轮对话已完成** ✅ (2026-10-05)
> 本计划定义从“智能管道”进化为“真正 AI Agent”的后续路线。

## 当前架构总结

```
用户 → 上传音视频 → LangGraph StateGraph (6 节点) → ASR → Agent Loop (Function Calling) → Markdown 报告
```

- **已完成**: 4 种输入源统一、LangGraph StateGraph、进度回调 + step_idx、前端 7 步详细流程 + 折叠详情面板
- **核心差距**: 无 Tool Use、无多轮对话、无记忆、无上下文工程、单用户、硬编码路由

---

## 总览：4 个阶段 · 12 个里程碑

| 阶段 | 里程碑 | 优先级 | 面试价值 | 工作量 |
|------|--------|--------|----------|--------|
| **Phase 1** Agent 核心能力 | M1 Function Calling | ★★★ | ★★★★★ | 中 |
| | M2 多轮对话 | ★★★ | ★★★★★ | 中 |
| | M3 上下文工程 | ★★☆ | ★★★★☆ | 小 |
| **Phase 2** 记忆系统 | M4 短期记忆 (会话级) | ★★☆ | ★★★★☆ | 小 |
| | M5 长期记忆 (向量检索) | ★★☆ | ★★★★☆ | 中 |
| | M6 用户偏好记忆 | ★☆☆ | ★★★☆☆ | 小 |
| **Phase 3** 可扩展架构 | M7 Skills 可插拔架构 | ★★☆ | ★★★☆☆ | 中 |
| | M8 MCP Server | ★☆☆ | ★★★★☆ | 中 |
| | M9 Streaming 流式输出 | ★★☆ | ★★★☆☆ | 中 |
| **Phase 4** 生产化 | M10 多用户管理 | ★★☆ | ★★☆☆☆ | 大 |
| | M11 可观测性 (Tracing) | ★★☆ | ★★★☆☆ | 中 |
| | M12 评估框架 (Eval) | ★☆☆ | ★★★☆☆ | 中 |

---

## Phase 1: Agent 核心能力

### M1 — Function Calling / Tool Use

**目标**: 将硬编码的 ASR/分类/摘要调用改为 LLM 自主决策的 Tool 调用，让 Agent "自己决定"何时转写、何时分类、何时生成摘要。

**当前问题**:
- `LLMClient.chat()` 只支持纯文本对话，不支持 `tools` 参数
- 分类器 (`classifier.py`) 是独立的 LLM 调用，不是 Agent 自主决策
- 路由逻辑 (`route_by_type`) 是硬编码的 if/elif

**改造方案**:

```
                    ┌─────────────────────────────────┐
                    │     Agent Loop (LangGraph)       │
                    │                                  │
  用户输入 ────────▶│  LLM (with tools) ──┐            │
                    │       │              │            │
                    │       ▼              │            │
                    │   Tool Call? ──yes──▶ 执行 Tool   │
                    │       │              │            │
                    │       no             │            │
                    │       ▼              │            │
                    │   返回最终结果 ◀─────┘            │
                    └─────────────────────────────────┘
```

**Tools 定义**:

| Tool 名称 | 功能 | 对应现有代码 |
|-----------|------|-------------|
| `transcribe_audio` | ASR 语音转写 | `asr_client.py` |
| `classify_recording` | 录音类型分类 | `classifier.py` |
| `generate_meeting_summary` | 生成会议纪要 | `pipeline.py` meeting 分支 |
| `generate_interview_analysis` | 生成面试分析 | `pipeline.py` interview 分支 |
| `generate_general_summary` | 生成通用总结 | `pipeline.py` general 分支 |
| `search_past_meetings` | 搜索历史会议 (Phase 2) | 新增 |
| `get_video_info` | 获取视频元信息 | `video_client.py` |

**改动文件**:
- `agent/llm_client.py` — 添加 `chat_with_tools()` 方法，支持 OpenAI tools 协议
- `agent/tools.py` — **新建**，定义所有 Tool 的 schema 和实现
- `agent/graph.py` — 重构 `classify` 和 `analyze_dispatch` 节点为 Tool 调用模式
- `agent/prompts.py` — 更新 system prompt，描述可用 Tools

**关键设计决策**:
- 使用 DashScope 的 OpenAI 兼容 `tools` 接口（`tool_choice="auto"`）
- Tool schema 使用 JSON Schema 格式
- 保留 LangGraph 作为编排层，Tool 调用是节点内部行为
- 分类器从"独立 LLM 调用"变为"Agent 可选 Tool"

---

### M2 — 多轮对话

**目标**: 用户完成一次分析后，可以基于结果继续追问（"帮我提取待办事项"、"面试者技术水平如何"、"翻译成英文"）。

**当前问题**:
- 完全单轮：上传 → 分析 → 返回结果，交互结束
- 无对话历史概念
- `LLMClient` 只支持 `messages: [system, user]` 两轮

**改造方案**:

```
前端 Chat UI:
  ┌─────────────────────────────────────┐
  │ 📋 会议纪要已生成！                    │
  │ ┌─ 折叠面板: 处理详情 ──────────────┐ │
  │ └─────────────────────────────────┘ │
  │                                      │
  │ 💬 对话区域                           │
  │ 🧑 帮我提取所有待办事项               │
  │ 🤖 根据会议内容，以下是待办事项...     │
  │ 🧑 第二条能展开说说吗？               │
  │ 🤖 当然，关于XX项目的待办...           │
  │ ┌──────────────────────────────┐     │
  │ │ 输入消息...              发送 │     │
  │ └──────────────────────────────┘     │
  └─────────────────────────────────────┘
```

**后端设计**:
- `server.py` 添加 `/api/chat/{task_id}` 接口，接收用户追问
- 维护 `conversation_history: list[dict]` 在内存中（按 task_id）
- 追问时：将转写文本 + 分析结果 + 历史对话组装为 messages，调用 LLM
- LangGraph 新增 `chat` 节点，支持多轮对话模式

**State 扩展**:
```python
class AgentState(TypedDict):
    # ... 现有字段 ...
    conversation_history: list[dict]  # 新增: 对话历史
    chat_mode: bool                   # 新增: 是否进入对话模式
```

**改动文件**:
- `server.py` — 添加 `/api/chat/{task_id}` 接口 + 对话历史存储
- `agent/graph.py` — 添加 `chat` 节点
- `agent/llm_client.py` — `chat()` 支持传入完整 `messages` 列表
- `frontend/index.html` — 添加 Chat UI（消息列表 + 输入框）

---

### M3 — 上下文工程

**目标**: 精确控制输入 LLM 的上下文内容，在 token 限制内最大化信息密度。

**当前问题**:
- 分类器截取前 3000 字符，无 token 精确计算
- 长音频转写文本可能超出 LLM 上下文窗口
- 无上下文压缩机制
- system prompt 硬编码，无动态组装

**改造方案**:

1. **Token 精确计算**: 引入 `tiktoken` 库，所有 prompt 构建基于 token 数而非字符数
2. **上下文分层**:
   ```
   ┌─ System Prompt (动态组装) ─────────────┐
   │  - 角色定义                             │
   │  - 可用 Tools 描述                      │
   │  - 用户偏好 (Phase 2)                   │
   ├─ Context (检索 + 压缩) ────────────────┤
   │  - 转写文本摘要 (长文本自动压缩)         │
   │  - 相关历史会议 (RAG 检索)               │
   │  - 当前对话历史                          │
   ├─ User Message ─────────────────────────┤
   │  - 用户当前输入                          │
   └────────────────────────────────────────┘
   ```
3. **长文本处理**: 转写文本 > 阈值时自动生成摘要，用摘要替代原文
4. **上下文窗口适配**: 根据模型实际上下文大小动态调整

**改动文件**:
- `agent/context.py` — **新建**，上下文管理器（token 计算、压缩、组装）
- `agent/llm_client.py` — 集成 context manager
- `requirements.txt` — 添加 `tiktoken`

---

## Phase 2: 记忆系统

### M4 — 短期记忆 (会话级)

**目标**: 在同一次分析会话中，记住中间结果和用户偏好，支持"刚才那个"、"再分析一下"等指代。

**实现**:
- 会话上下文窗口：保留最近 N 轮对话
- 任务结果缓存：分析完成后，结果作为上下文供追问使用
- 滑动窗口：超过 N 轮时，压缩早期对话为摘要

**改动文件**:
- `agent/memory.py` — **新建**，短期记忆管理
- `server.py` — 会话状态管理

### M5 — 长期记忆 (向量检索)

**目标**: 跨任务记忆，支持"上次关于XX项目的会议说了什么"、"对比最近三次会议的决定"。

**实现**:
```
分析完成 → 文本分块 → Embedding → 向量数据库
                                         │
用户追问 "对比上次会议" → 查询向量库 → 检索相关片段 → 注入上下文
```

**技术选型**:
- 向量数据库: ChromaDB (轻量、本地、Python 原生)
- Embedding: DashScope text-embedding-v3 (与现有 ASR/LLM 同一平台)
- 分块策略: 按段落/章节分块，保留元数据 (task_id, 时间, 类型)

**改动文件**:
- `agent/vector_store.py` — **新建**，向量存储 + 检索
- `agent/tools.py` — 添加 `search_past_meetings` Tool
- `requirements.txt` — 添加 `chromadb`
- `server.py` — 分析完成后自动存入向量库

### M6 — 用户偏好记忆

**目标**: 记住用户的输出格式偏好、常用模型、关注重点等。

**实现**:
- 偏好存储: JSON 文件 (轻量) 或 SQLite
- 偏好类型: 输出语言、详细程度、关注维度、默认模型
- 自动学习: 从用户追问中提取偏好信号

**改动文件**:
- `agent/preferences.py` — **新建**
- `agent/context.py` — 动态注入用户偏好到 system prompt

---

## Phase 3: 可扩展架构

### M7 — Skills 可插拔架构

**目标**: 将 meeting/interview/general 三种分析策略抽象为 Skill 插件，支持热插拔扩展。

**当前问题**:
- `analyze_dispatch` 中 if/elif 硬编码 3 种策略
- 新增分析类型需修改核心代码

**改造方案**:
```python
# agent/skills/base.py
class AnalysisSkill(Protocol):
    name: str
    description: str
    trigger_condition: str  # 自然语言描述，LLM 判断是否触发

    async def analyze(self, transcript: str, **kwargs) -> str: ...

# agent/skills/meeting.py — MeetingSkill
# agent/skills/interview.py — InterviewSkill
# agent/skills/general.py — GeneralSkill
# agent/skills/lecture.py — LectureSkill (新增示例)
```

**Skill 注册表**:
```python
# agent/skills/registry.py
class SkillRegistry:
    skills: dict[str, AnalysisSkill]

    def register(self, skill: AnalysisSkill): ...
    def select_skill(self, description: str) -> AnalysisSkill: ...
    # LLM 根据 description 自动选择 Skill
```

**改动文件**:
- `agent/skills/` — **新建目录**，每个策略一个文件
- `agent/skills/registry.py` — Skill 注册和选择
- `agent/graph.py` — `analyze_dispatch` 改为从 registry 动态加载

### M8 — MCP Server

**目标**: 将核心能力暴露为标准 MCP (Model Context Protocol) 服务，让其他 AI 应用（如 Claude Desktop、Cursor）直接调用。

**暴露的 Tools**:
```
mcp_meeting_agent/
├── transcribe(url_or_file)      → ASR 转写文本
├── analyze(url_or_file)         → 完整分析流程
├── summarize(task_id)           → 获取已有分析结果
├── search(query)                → 搜索历史会议
└── chat(task_id, message)       → 多轮对话
```

**技术选型**:
- `mcp` Python SDK (官方)
- 独立进程运行，通过 stdio/SSE 通信

**改动文件**:
- `mcp_server/` — **新建目录**
- `mcp_server/server.py` — MCP Server 定义
- `mcp_server/tools.py` — Tool 实现 (复用 agent/ 模块)

### M9 — Streaming 流式输出

**目标**: LLM 分析结果流式输出到前端，用户无需等待完整生成。

**实现**:
- 后端: SSE (Server-Sent Events) 或 WebSocket
- 前端: 逐字渲染 Markdown
- LangGraph: 利用 `astream_events` 获取节点级流

**改动文件**:
- `server.py` — 添加 `/api/stream/{task_id}` SSE 端点
- `agent/graph.py` — 使用 `graph.astream_events()`
- `frontend/index.html` — SSE 接收 + 流式渲染

---

## Phase 4: 生产化

### M10 — 多用户管理

**目标**: 支持多用户独立使用，任务隔离，配额管理。

**实现**:
- 用户认证: API Key 或简单 JWT
- 任务隔离: 每个用户独立的 uploads/output 目录
- 配额限制: 每日/每小时任务数限制
- 数据库: SQLite (轻量) 存储用户信息和任务记录

**改动文件**:
- `agent/auth.py` — **新建**，认证和授权
- `server.py` — 添加认证中间件
- `database.py` — **新建**，SQLite 数据层

### M11 — 可观测性 (Tracing)

**目标**: 全链路追踪，方便调试和优化。

**技术选型**: LangSmith (LangChain 官方) 或 OpenTelemetry

**实现**:
- 每个节点自动记录: 输入/输出、耗时、token 用量
- 可视化: 执行轨迹图、耗时分布
- 告警: 异常耗时自动标记

**改动文件**:
- `agent/tracing.py` — **新建**
- `agent/graph.py` — 集成 tracing callbacks
- `requirements.txt` — 添加 `langsmith` 或 `opentelemetry-sdk`

### M12 — 评估框架 (Eval)

**目标**: 量化评估分析质量，支持回归测试。

**实现**:
- 评估维度: 摘要完整性、事实准确性、格式规范性
- 评估方法: LLM-as-Judge + 人工标注对比
- 基准数据集: 准备 10-20 个标注样本
- CI 集成: 每次代码变更自动运行评估

**改动文件**:
- `eval/` — **新建目录**
- `eval/dataset.json` — 标注数据集
- `eval/run_eval.py` — 评估脚本
- `eval/judge.py` — LLM-as-Judge 实现

---

## 补充建议 (用户未提及但值得做)

### A. 错误恢复与重试机制

**问题**: 当前 ASR/LLM 失败后整个任务直接失败。
**方案**: LangGraph 节点级别重试 + 降级策略。
```python
builder.add_node("asr_transcribe", asr_transcribe, retry=RetryPolicy(max_attempts=3))
```

### B. 并行节点执行

**问题**: 当前所有节点串行执行，部分节点可并行。
**方案**: 面试分析中 question.md 和 analyze.md 可并行生成；ASR 和文本预处理可流水线并行。
**改动**: LangGraph 的 `Send` API 或 fan-out/fan-in 模式。

### C. 输入验证与安全

**问题**: 无文件大小限制、无 URL 安全校验、无速率限制。
**方案**: 添加输入验证层，防止恶意输入。

---

## 实施建议

### 推荐实施顺序

```
Phase 1 (1-2 周)          Phase 2 (1-2 周)          Phase 3 (1 周)           Phase 4 (持续)
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ M1 Function Call │────▶│ M4 短期记忆       │────▶│ M7 Skills 架构   │────▶│ M10 多用户       │
│ M2 多轮对话       │     │ M5 长期记忆       │     │ M8 MCP Server    │     │ M11 Tracing      │
│ M3 上下文工程     │     │ M6 用户偏好       │     │ M9 Streaming     │     │ M12 Eval         │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

### 每个里程碑的标准流程

1. **设计**: 接口定义 + 数据流图
2. **实现**: 后端 → 前端 → 集成
3. **测试**: 单元测试 + 端到端测试
4. **文档**: 更新 README + 流程图

### 面试价值最大化策略

| 面试考点 | 对应里程碑 | 关键话术 |
|---------|-----------|---------|
| Function Calling | M1 | "Agent 通过 Tool Use 自主决策调用 ASR/分类/摘要" |
| 多轮对话 | M2 | "基于转写文本和分析结果的上下文追问" |
| 上下文工程 | M3 | "tiktoken 精确计算 + 动态上下文压缩" |
| RAG | M5 | "ChromaDB 向量检索历史会议，检索增强生成" |
| LangGraph | p01 (已完成) | "StateGraph 条件路由，可视化执行轨迹" |
| MCP | M8 | "标准 MCP 协议暴露能力，跨应用互操作" |
| Agent 架构 | M7 | "Skill 可插拔架构，Protocol 协议驱动" |

---

## 文件变更预览

Phase 1 完成后新增文件:
```
agent/
├── tools.py              # M1: Tool 定义和实现
├── context.py            # M3: 上下文管理器
├── memory.py             # M4: 短期记忆
├── vector_store.py       # M5: 向量存储
├── preferences.py        # M6: 用户偏好
├── skills/               # M7: Skills 架构
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── meeting.py
│   ├── interview.py
│   └── general.py
├── tracing.py            # M11: 可观测性
mcp_server/               # M8: MCP Server
├── server.py
└── tools.py
eval/                     # M12: 评估框架
├── dataset.json
├── run_eval.py
└── judge.py
```
