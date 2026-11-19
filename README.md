# 会议助手智能体（Meeting Agent）

基于 ASR、LLM 和 LangGraph 的音视频分析项目，支持本地文件和在线链接，生成会议纪要、面试分析或通用内容总结。提供 Typer 命令行和 FastAPI Web 服务，浏览器界面位于 `frontend/index.html`，无需单独构建前端。

## 当前能力

- 音频转写：通过阿里云百炼 ASR 获取文本、时间戳和说话人信息，清洗语气词并合并短句。
- 视频处理：使用 ffmpeg 提取本地视频音频，使用 yt-dlp 处理在线链接，并提供字幕提取逻辑；站点可用性取决于权限、cookies 和 yt-dlp 的支持情况。
- 自动分析：通过工具调用识别 meeting / interview / general 类型，生成对应报告；工具调用异常时支持传统分类、分析路径降级。
- 长文本处理：分段分析、重叠上下文、token 预算管理。
- Web 交互：文件上传、链接提交、模型选择、任务进度、SSE 流式输出、结果预览和下载。
- 对话与记忆：围绕单个任务追问、对话压缩和里程碑；可选 ChromaDB 长期记忆、Embedding 和 Rerank，以及跨任务问答。
- 运维辅助：ASR / 总结文件缓存、按 IP 的每日任务配额、可选 LangSmith 追踪。

## 架构与处理流程

CLI 和 Web 后台任务共用 `agent/graph.py` 的状态图：

```text
输入（本地音频 / 本地视频 / 音频 URL / 视频 URL）
  → extract_audio：处理来源、提取音频或字幕
  → preprocess：本地音频预处理
  → asr_transcribe：语音转写及缓存
  → text_clean：文本清洗
  → agent_loop：分类与分析工具调用，异常时降级
  → format_output：格式化并保存报告
```

本地音频预处理使用单声道、默认 16 kHz WAV；远程音频可直接交给 ASR。Web 任务由后台线程执行，通过状态查询和 SSE 向页面返回进度。分析结果可按配置进入向量索引，供后续检索使用。

## Ubuntu 部署与启动

部署目录为 `~/projects/meeting_agent`，运行账户为 `zane`。现有 Ubuntu 已安装 uv、FFmpeg 和 FFprobe。新环境可按 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)安装 uv，系统命令可用 `sudo apt install ffmpeg curl iproute2 util-linux` 补齐。项目通过 `.python-version`、`pyproject.toml` 和 `uv.lock` 锁定 CPython 3.12.14 与依赖。

从 Mac 同步源码时排除配置密钥和运行数据，然后登录 Ubuntu：

```bash
ssh ubuntu 'mkdir -p ~/projects/meeting_agent'
rsync -av --exclude=.git --exclude=.env --exclude=.venv --exclude=data/ \
  --exclude=logs/ --exclude=output/ --exclude=uploads/ --exclude=__pycache__/ \
  --exclude=.idea --exclude=.qoder --exclude=.DS_Store --exclude='*.log' \
  ./ ubuntu:projects/meeting_agent/
ssh ubuntu
cd ~/projects/meeting_agent
test -f .env || install -m 600 .env.example .env
./start.sh
```

首次启动会执行 `uv sync --locked`，自动安装匹配的 Python 和依赖，随后通过用户级 systemd 启动 Web 服务。脚本打印依赖同步、启动日志、HTTP 健康检查和服务状态。若部署账户未启用 linger，可执行 `loginctl enable-linger "$USER"`，使服务在 SSH 退出后继续运行并随系统启动。更新源码或修改配置后执行 `./start.sh restart`。

本次使用 uv 0.13.7。若 Ubuntu 直连下载缓慢且原有代理可用，可用 `HTTPS_PROXY=http://127.0.0.1:17890 ./start.sh` 仅加速启动时的依赖下载。

录音转写和分析前，在 Ubuntu 的 `.env` 中填写百炼账号的 `DASHSCOPE_API_KEY`。LLM 默认复用这个密钥。使用其他兼容服务时按需设置 `LLM_API_KEY`、`LLM_BASE_URL` 和 JSON 数组格式的 `LLM_MODELS`。`ASR_MODELS` 也使用 JSON 数组。旧配置中的单数 `LLM_MODEL` 应迁移成 `LLM_MODELS`。在线视频需要登录态时，可将 Netscape 格式 cookies 放到 `data/cookies/`。配置和 cookies 留在部署账户内。

### 从 Mac 访问

打开 [Meeting Agent](http://127.0.0.1:19190/)。Ubuntu 服务绑定 `127.0.0.1:19090`。本次在 Mac 当前登录会话通过 `launchctl` 保持 SSH 转发，可用 `launchctl remove local.meeting-agent.forward` 关闭。转发断开后，在 Mac 终端重新运行：

```bash
ssh -fNT -o ExitOnForwardFailure=yes -L 127.0.0.1:19190:127.0.0.1:19090 ubuntu
```

健康检查位于 [`/api/health`](http://127.0.0.1:19190/api/health)。其中 `api_key_configured` 表示已填入百炼密钥，模型调用需以实际上传任务确认。

### 状态与日志

| 命令 | 用途 |
| --- | --- |
| `./start.sh` | 启动并等待健康检查 |
| `./start.sh status` | 查询 systemd 状态和健康接口 |
| `./start.sh logs` | 查看最近 60 行启动、访问和应用日志 |
| `./start.sh logs --follow` | 持续跟踪日志 |
| `./start.sh restart` / `./start.sh stop` | 重启 / 停止 |
| `logs/server_YYYY-MM-DD.log` | 应用 DEBUG 日志，按天轮转，保留 7 天 |

在 Mac 本地开发时，可从项目根目录运行 `uv sync --locked`，再运行 `uv run --locked uvicorn server:app --host 127.0.0.1 --port 9090`。

## 使用方式

### 命令行

```bash
uv run --locked python main.py analyze /path/to/recording.wav
uv run --locked python main.py analyze "https://example.com/audio.mp3" -v
uv run --locked python main.py video /path/to/video.mp4
uv run --locked python main.py video "https://www.bilibili.com/video/BVxxxx/"
uv run --locked python main.py transcript-only /path/to/recording.wav
```

`analyze` 和 `video` 支持 `-v` / `--verbose`；当前没有 `--output json` 选项。

当前 CLI 的 `_run_graph_pipeline` 未将状态图返回的 `result_path` 写回其返回字典，因此报告可能已保存，但终端仍提示“未找到结果文件”；可检查 `output/summary/`，或使用 Web 界面查看结果。

## 常用 API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST | `/api/upload` | multipart 文件上传；`file` 为文件字段，模型参数为查询参数 |
| POST | `/api/submit-url` | JSON 提交 `url`，可带 `asr_model`、`llm_model` |
| GET | `/api/status/{task_id}` | 任务状态、进度和元数据 |
| GET | `/api/stream/{task_id}` | 任务 SSE 输出 |
| GET | `/api/result/{task_id}` | 获取报告 |
| GET | `/api/result/{task_id}/download` | 下载报告 |
| GET | `/api/asr/{task_id}` | 获取转写文本 |
| GET | `/api/asr/{task_id}/download` | 下载转写文本 |
| POST | `/api/chat/{task_id}` | 任务对话，JSON 字段 `message` |
| POST | `/api/chat/{task_id}/stream` | 流式任务对话 |
| GET | `/api/chat/{task_id}/history` | 任务对话历史 |
| POST | `/api/chat/{task_id}/milestone` | 标记对话里程碑 |
| POST | `/api/global-chat`、`/api/global-chat/stream` | 全局问答及流式问答 |
| GET / DELETE | `/api/global-chat/history` | 获取 / 清空指定会话历史 |
| GET | `/api/models`、`/api/health`、`/api/quota` | 模型、健康检查和配额 |
| GET | `/api/vector-store/stats` | 向量库统计 |

提交链接示例（服务启动后执行）：

```bash
curl -X POST http://localhost:9090/api/submit-url \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/audio.mp3"}'
```

## 配置参考

配置由 `config.py` 的 `Settings` 定义，从环境变量和根目录 `.env` 读取。

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | 空 | ASR 密钥，也是 LLM 默认密钥来源 |
| `LLM_API_KEY` | 复用百炼密钥 | 独立 LLM 密钥 |
| `LLM_BASE_URL` | 百炼兼容接口地址 | LLM 服务地址 |
| `LLM_MODELS` | 首项 `qwen3.7-flash-2026-07-15` | LLM 模型优先级列表 |
| `ASR_MODELS` | 首项 `paraformer-v2` | ASR 模型优先级列表 |
| `ASR_SAMPLE_RATE` | `16000` | 预处理采样率 |
| `MAX_TOKENS_PER_CHUNK` | `6000` | 文本分片 token 预算 |
| `MIN_SENTENCE_LENGTH` | `10` | 短句合并阈值 |
| `OUTPUT_DIR` | `./output` | 转写和报告目录 |
| `ENABLE_VECTOR_STORE` | `false` | 开启长期记忆索引与检索 |
| `EMBEDDING_MODEL` | `qwen3.7-text-embedding-flash` | 向量模型 |
| `RERANK_MODEL` | `qwen3.7-text-rerank` | 重排模型 |
| `VECTOR_STORE_DIR` | `./data/vector_store` | ChromaDB 持久化位置 |
| `DAILY_TASK_LIMIT` | `5` | 每个 IP 的每日任务上限 |
| `QUOTA_WHITELIST_IPS` | `[]` | 不受配额限制的 IP，JSON 数组 |
| `QUOTA_DATA_DIR` | `./data` | 配额数据目录 |
| `LANGCHAIN_TRACING_V2` | `false` | 开启 LangSmith 追踪 |
| `LANGSMITH_API_KEY` | 空 | LangSmith 密钥 |
| `LANGCHAIN_PROJECT` | `meeting_agent` | 追踪项目名 |
| `LANGSMITH_ENDPOINT` | `https://api.smith.langchain.com` | 追踪服务地址 |

开启向量检索前需确认 Embedding 和 Rerank 模型可访问。服务端配额按 IP 统计，不是登录账号配额；反向代理部署时应控制可信的转发头。

## 项目目录

```text
meeting_agent/
├── main.py                 # Typer CLI
├── server.py               # FastAPI、后台任务、SSE、对话与检索接口
├── config.py               # Pydantic Settings
├── pyproject.toml、uv.lock     # Python 版本约束和锁定依赖
├── start.sh、deploy/         # Ubuntu 启动脚本和 systemd 用户服务
├── frontend/index.html     # Web 界面
├── agent/
│   ├── graph.py            # LangGraph 状态与节点编排
│   ├── tools.py            # 分类、会议 / 面试 / 通用分析工具
│   ├── pipeline.py         # 分析与 Markdown 格式化实现
│   ├── asr_client.py       # ASR 上传、轮询和结果解析
│   ├── llm_client.py       # 兼容接口、工具调用、流式输出
│   ├── classifier.py       # 录音分类
│   ├── prompts.py          # 提示词
│   ├── text_processor.py   # 清洗、分段与重叠窗口
│   ├── video_client.py     # 视频下载、音频和字幕提取
│   ├── cache.py            # 转写和报告缓存
│   ├── context.py          # token 预算与上下文组装
│   ├── memory.py           # 对话记忆、压缩和里程碑
│   ├── chunker.py          # 父子分块
│   ├── embedding.py        # 向量模型客户端
│   ├── reranker.py         # 检索重排客户端
│   ├── vector_store.py     # ChromaDB 索引与检索
│   ├── quota.py            # 按 IP 配额及持久化
│   └── tracing.py          # LangSmith 配置
├── models/schemas.py       # 转写、会议结果等数据模型
├── utils/audio.py          # 音频转换工具
├── docs/graph.mmd          # 流程图源文件，实际流程以 graph.py 为准
├── docs/graph.png
├── test_stream.py          # 手工 SSE 调试脚本
└── test_stream.html        # 浏览器流式调试页面
```

## 输出与状态

```text
output/
├── asr/                          # 转写缓存：{name}_{asr_model}.txt
└── summary/
    ├── meeting/                  # 会议纪要
    ├── interview/                # 每份报告目录下 question.md、analyze.md
    ├── video/                    # 视频总结
    └── other/                    # 通用总结
uploads/                          # Web 上传文件，按任务 ID 分目录
 data/                            # cookies、配额数据、可选向量库
```

报告通常按来源名称和所用模型命名，具体规则见 `agent/cache.py` 和 `server.py`。任务状态和对话历史保存在进程内存中，服务重启后不会自动恢复；报告、转写缓存、配额数据及可选向量库保存在磁盘。当前没有完整的账号认证和任务隔离体系，对外部署前需补充访问控制。

## 验证与开发现状

运行 `uv run --locked python -m unittest discover -s tests -v`，测试启动脚本的依赖同步、状态检查、端口冲突和失败处理。`test_stream.py` 是独立的 SSE 手工调试脚本，需要提供真实任务 ID。

配置密钥后，可上传一段短录音，检查任务进度、转写和报告下载、追问及重复处理时的缓存命中。

后续的 PostgreSQL 存储、分层缓存、历史页面、RabbitMQ 和 GPU 调度见 [实施计划](docs/persistence-queue-gpu-plan.md)。

会议历史问答的 Baseline、目标版本、NL2SQL 和事件图谱选型见 [开发计划](docs/meeting-knowledge-query-plan.md) 与 [架构图册](docs/architecture/README.md)。

## License

[MIT](LICENSE)

### 启用 PostgreSQL 历史

在 PostgreSQL 创建 `meeting_agent` 数据库及专用迁移/运行账号。迁移账号拥有数据库或可在其中创建 `meeting` schema。运行账号需要该 schema 的 USAGE、业务表 DML 和序列 USAGE。用私有配置设置迁移连接 `MIGRATION_DATABASE_URL`，执行 `uv run --locked alembic upgrade head`，再在项目 `.env` 设置运行账号的 `DATABASE_URL` 并重启。网页的「分析历史」可筛选任务并查看媒体、转写、报告和题目。

旧文件先执行 `uv run --locked python -m persistence.legacy --root output` 查看导入数量，确认后加 `--apply`。仅唯一匹配的旧转写会关联报告，未知来源时间仍留空。

隔离库迁移后设置 `TEST_DATABASE_URL` 可运行实际 PostgreSQL 验收。

### 管理阶段缓存

完成数据库迁移后打开「分析历史 → 缓存管理」。工作区有来源、预处理、ASR、清洗、分类、面试分段和最终分析七层，可各自关闭读取或写入，按任务创建时锁定策略。已完成的报告和转写仍可通过历史访问。失效操作撤销复用资格，缓存文件由后续磁盘回收流程处理。

### 启用 RabbitMQ 与批量任务

创建项目 RabbitMQ vhost，允许项目账号声明 `meeting.tasks`、`meeting.dlx` 和队列。运行数据库迁移，配置 `AMQP_URL` 以及可选的 `AMQP_WORKER_URL`。安装 `deploy/meeting-agent-dispatcher.service` 和 `deploy/meeting-agent-worker@.service` 到 `~/.config/systemd/user/`，执行 `systemctl --user daemon-reload`，启用 dispatcher 及 `meeting-agent-worker@io.service`、`@cpu.service`、`@cloud.service`。开发环境可分别运行 `uv run --locked python -m persistence.broker dispatcher|io|cpu|cloud`。

页面「批量任务」支持最多 20 项混合提交，保留原有单项 API。RabbitMQ 不可用时已承诺的任务保存在 PostgreSQL，并在恢复后派发。失败任务可从已保存的上一阶段续跑。

### 启用共享 GPU 和本地 ASR

在同一 Ubuntu PostgreSQL 实例建立独立 `gpu_ops` 数据库，并授权受管项目账号使用同一数据库。设置 `GPU_DATABASE_URL`，按需指定 `GPU_DEVICE_UUID`。先以迁移账号运行 `uv run --locked python -m gpu_ops init`。在小样本冷、热启动校准后登记显存预算，例如 `python -m gpu_ops profile whisper-small 3072 --margin 1024`。预算为预计新增显存，另加安全余量。`--exclusive` 要求无其他 CUDA 计算进程。

配置 `LOCAL_ASR_PYTHON` 和预下载的 `LOCAL_ASR_MODEL` 后，可在页面选择 `local-whisper`。启用 `meeting-agent-worker@gpu.service` 和 `meeting-agent-gpu-watch.service`。等待状态会释放 RabbitMQ 消费槽，任务由数据库按间隔重新派发。用 `python -m gpu_ops status`、批量任务页或 `/api/gpu/metrics` 查询显存、持有者和等待告警。watch 日志仅在告警升级或恢复时记录状态变化。

其他项目可使用同一调度器执行：

```bash
uv run --locked python -m gpu_ops run --project rag_law --profile rag-small -- /path/to/python /path/to/task.py
printf '只回答：就绪\n' | uv run --locked python -m gpu_ops run --project ollama --profile ollama-small -- .venv/bin/python -m gpu_ops.ollama_request MODEL
```

每个命令先按实测需求注册 profile。Ollama 建议单模型、单并发，受管请求限制上下文并在结束时卸载模型。其他项目直接启动 GPU 时，系统只能观测它们的实际占用，在无法满足预算时将本地任务留在队列。稳定性观察达到计划标准后再决定是否默认提供 GPU 入口。
