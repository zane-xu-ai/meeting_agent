# Meeting Agent 后续实施计划

2026-11-17 · 方案待实施

## 1. 现状和架构选择

- Web 当前通过后台线程执行 LangGraph，任务状态、SSE 队列和对话依赖进程内存。转写、报告及面试题单落盘。旧缓存依赖文件名和模型名，部分读取允许前缀匹配，同名不同内容可能误命中。缓存转写也可能被再次清洗。
- Ubuntu 实测已有 PostgreSQL 和 RabbitMQ 容器，两者当时均停止。Ollama 在 `127.0.0.1:11434` 运行，检查时无已加载模型。RTX 3060 Ti 共 8192 MiB，采样时空闲 5365 MiB，`rag_law` 两个进程合用约 2448 MiB。后续调度以实时采样为准。
- **PostgreSQL 保存业务记录、任务状态、转写和分析正文。Ubuntu 文件系统保存音视频及大文件。RabbitMQ 派发可恢复的阶段任务。** 百炼等远程 API 及常规 FFmpeg 任务走各自队列，本地推理另设 GPU 准入。首版暂缓引入 Redis 和对象存储。已有 pgvector 可在后续承担向量索引。

## 2. SQL 存储及文件布局

复用 Ubuntu 的 PostgreSQL 17 容器，回环地址 `127.0.0.1:5432`。创建数据库 **`meeting_agent`**，schema **`meeting`**。使用项目专用的 `meeting_agent_migrator`、`meeting_agent_app`，迁移账户拥有 schema，运行账户按表授予必要的 DML。跨项目访问须独立授权。用 SQLAlchemy 2、psycopg 3、Alembic，全部纳入 uv 锁文件。管理员在现有实例上执行显式建库、建角色和授权，Alembic 管理项目对象。运行时连接串放服务器私有配置中。

表名均位于 `meeting`。主键默认 UUID，时间统一为 `timestamptz`，音视频时间戳和时长统一为整型毫秒。等待状态用文本枚举约束，评分用数值列，原生报文和版本化配置用 JSONB。各表记录 `workspace_id` 或经外键归属到工作区。

| 表 | 关键字段及用途 |
| --- | --- |
| `workspaces`、`workspace_members` | 工作区及成员角色，区分可查看、提交、管理缓存和删除的权限。初期给现有个人部署建立一个工作区。 |
| `media_sources` | `id, workspace_id, source_kind, original_filename, original_url, canonical_url, platform, platform_item_id, title, description, uploader, published_at, source_created_at, source_modified_at, timestamp_provenance, first_seen_at, last_seen_at, created_at, updated_at`。保留原入口、显示标题和采集来源。带临时授权的 URL 分离保存，列表和日志仅展示脱敏版本。 |
| `media_versions` | `id, source_id, file_artifact_id, content_sha256, size_bytes, mime_type, container, audio_codec, video_codec, duration_ms, sample_rate_hz, channels, width, height, fetched_at, etag, http_last_modified, source_metadata_snapshot, probe_json`。`source_metadata_snapshot` 保存该次采集的标题、文件名、解析器版本及来源时间。同一 URL 的内容更新后新增版本，原任务仍引用原版本。上传尚未获取原内容时，先登记采集阶段，待完整哈希后绑定版本。 |
| `artifacts` | `id, workspace_id, sha256, storage_key, bytes, mime_type, kind, status, created_at`。指向原始文件、下载媒体、PCM、字幕快照或导出物。按工作区和哈希复用物理文件。用引用关系和孤儿扫描决定回收。 |
| `submission_batches` | `id, workspace_id, submitted_by, idempotency_key, request_hash, created_at`。批次进度按子任务状态聚合，支持某条输入失败而其他条继续。 |
| `analysis_jobs` | `id, legacy_task_id, workspace_id, batch_id, source_id, media_version_id, submitted_by, state, progress, current_stage, priority, parent_job_id, idempotency_key, request_config, cache_policy_snapshot, queued_at, started_at, finished_at, cancel_requested_at, error_code, error_summary`。一次提交一个任务，重跑建立新行并关联上次任务。 |
| `stage_runs` | `id, job_id, stage, stage_unit, attempt, state, input_fingerprint, cache_entry_id, output_ref, provider, requested_model, actual_model, model_revision, provider_job_id, settings_snapshot, code_version, prompt_version, started_at, ended_at, heartbeat_at, lease_token, lease_expires_at, usage, error_code, error_summary`。记录分片、模型回退、缓存命中和失败阶段。约束 `(job_id, stage, stage_unit, attempt)` 唯一。 |
| `stage_outputs` | `id, stage_run_id, kind, ordinal, result_json, artifact_id, content_sha256, schema_version, validation_state, created_at`。存切分清单、批次题目和评分、分段摘要等，分片以序号和转写段引用定位。分析结果的业务字段落 `analyses`、`analysis_questions`。 |
| `transcripts`、`transcript_segments` | 转写表存 `media_version_id, stage_run_id, variant, parent_transcript_id, provider, model, language, text, format_version, segment_count, content_sha256`。段表存 `transcript_id, seq, start_ms, end_ms, speaker, text, confidence, words_json`。原始 ASR、字幕和清洗结果分版本保留。段序号唯一，清洗记录原始父版本。 |
| `classifications` | `id, transcript_id, stage_run_id, category, confidence, reason, origin, provider, model, prompt_version, created_at`。类别包括 meeting、interview、general、video、unknown。人工纠正新增版本并归属到任务，保留自动判断记录。 |
| `analyses` | `id, job_id, revision, transcript_id, classification_id, stage_run_id, category, title, summary_text, overall_score, result_json, report_md, output_schema_version, actual_model, created_at`。JSONB 按类别放章节、决策、行动项、参与者、关键词、评分、建议等。Markdown 用于原样复现和下载。`(job_id, revision)` 唯一。 |
| `analysis_questions` | `analysis_id, ordinal, question_text, topic, start_ms, end_ms, evidence_segments, candidate_answer, recommended_answer, score, improvement_json`。时间戳不确定时留空，可附原始时间文本。编号和题单一一对应。展示和导出使用同一组记录。 |
| `job_events` | `job_id, seq, event_type, payload, created_at`。保存状态、步骤、缓存、等待、告警摘要及合并后的进度事件。唯一 `(job_id, seq)`，供 SSE 回放。高频预览片段可合并后设置较短保留期，完整报告留在 `analyses`。 |
| `cache_entries`、`cache_policies` | 缓存索引包含 `workspace_id, stage, key, status, result_ref, bytes, created_at, hit_count, last_hit_at, expires_at, invalidated_at`，引用字段按产物类型使用外键并约束唯一目标。策略表记录阶段、读写开关、TTL、修改者和修改时间。 |
| `outbox` | `id, job_id, stage_run_id, event_type, routing_key, payload, available_at, lease_until, publish_attempts, published_at, last_error_code`。任务变化及待派发事件同事务提交。 |
| `message_receipts`、`audit_events` | 记录已处置派发 ID，用唯一键防重；审计下载、删除、缓存切换、重跑和权限变更。凭据、全量媒体、原始模型思考及敏感请求体无须写入。 |

### 字段和索引规则

- 文件创建时间、最近修改时间和项目入库时间独立保存。普通浏览器上传可补传 `File.lastModified`，创建时间只能从受控导入或可靠源元数据获取。上传时间、Linux `ctime` 及 HTTP `Last-Modified` 各有不同语义，未知值留空并记录来源。`source_created_at` 和 `published_at` 分别表达文件来源创建时间和在线发布时间。
- 原始 URL 用于溯源；规范 URL 用于检索及版本匹配。保留影响资源身份的参数，识别到的跟踪参数可移除。记录重定向后可稳定访问的目标。元数据、ETag、Last-Modified 只用来判断是否重验，内容哈希用于判定具体版本。HTTP 304 可复用已验证快照。
- 常用索引为 `analysis_jobs(workspace_id, queued_at DESC, id DESC)`、`analysis_jobs(workspace_id, state, queued_at)`、`media_sources(workspace_id, canonical_url)`、`media_versions(source_id, content_sha256)`、`classifications(category, created_at)`、`outbox(available_at)` 未发布部分索引。转写段和题目以父 ID、序号索引。首版支持标题、文件名、URL、摘要和题目关键词；中文用 substring 并做样本性能验证，确认瓶颈后评估 pg_trgm、中文分词及 pgvector。
- 业务数据库中的转写、分类、报告、题目及任务关系是历史记录。缓存清理只使复用入口失效，历史详情和下载通过业务引用访问。分类纠正、版本变化或重跑保留历史来源，UI 可切换版本。
- 文件建议存于 `~/projects/meeting_agent/var/media/` 和 `var/derived/`，使用工作区、哈希分目录，文件权限 600。用流式上传写临时文件、校验空间和大小、同文件系统原子重命名，再提交 SQL 引用。失败孤儿由带宽限期的巡检回收。清理先标为删除中、检查业务及缓存引用，再做实际移除。通过有权限的 API 下载。每日备份数据库和文件清单及对应媒体，按一致快照做恢复演练。单机首版目标 RPO ≤24 小时、RTO ≤2 小时，恢复测试后确认。

## 3. 阶段缓存

统一键为 `SHA-256(规范序列化后的工作区、阶段、输入指纹和有效配置)`。模型阶段加入供应商、实际模型或版本、请求模型与回退策略、任务相关模型参数、提示词和输出结构版本。候选结果的模型必须仍满足当前策略，显式指定模型时只读对应模型的结果。读写使用短事务和唯一键。缓存构建使用有期限的持有者租约，同一键并发时等待既有构建或在超时后接管。下游用上游产物哈希入键，输入变化自然派生新键。

| 层 | 复用键的主要输入 | 建议位置和默认行为 |
| --- | --- | --- |
| L0 来源和下载 | URL、平台 ID、下载配置、登录态作用域、校验信息 | PG 管元数据及版本，音视频入文件存储。URL 元数据及下载快照默认可复用，6 小时后重验，临时直链在到期前失效。上传内容哈希归档用于媒体引用，可跨同工作区的不同任务引用。 |
| L1 音轨和预处理 | 原始字节哈希、音轨序号、FFmpeg 版本、声道、采样率和格式 | 已提取音轨和标准化 WAV 视同一层的两个产物，带哈希的文件存储。默认开，7 天或磁盘限额，活跃任务引用优先保护。临时探测、音频解码中间块按任务清理。 |
| L2 字幕或 ASR | 音轨或字幕快照哈希、ASR 参数、语言、说话人设置、供应商和模型 | PG 存原始转写及段，原始供应商附件可选存文件，清除临时签名 URL。默认开，建议缓存索引 90 天；成功任务转写随历史保留。空文本只有经明确无语音确认才可作为有效结果。 |
| L3 文本清洗与切分 | 原始转写哈希、清洗规则、分句、时间窗口、overlap、tokenizer 版本 | PG `transcripts` 存派生转写，`stage_outputs` 存分片清单。默认开，索引 30 天。分别存原始与清洗版本，老缓存导入时标识变体，避免反复清洗。 |
| L4 分类 | 清洗转写哈希、分类规则、模型、类别集合、提示词版本 | PG `classifications`，默认开，索引 90 天。人工纠正作为任务输入版本。 |
| L5 分段提问、摘要与评分 | 分片或题目及证据哈希、模型、提示词、输出结构 | PG `stage_outputs` 存验证完的阶段结果，默认开，索引 30 天。面试逐题分批及每批自己的校验状态作为断点。截断 JSON、错配题号和部分结果不得提升为可复用的完整分析。 |
| L6 最终分析与渲染 | 分类、段结果及合并规则哈希，渲染额外绑定模板版本 | PG `analyses` 和 `analysis_questions`，默认开，索引 90 天。相同正文可重新套模板。一次新任务即使命中仍留独立历史和引用。 |
| L7 检索索引 | 业务内容哈希、切片、编码器版本和模型 | 后续启用，索引可重建。保留现有 Chroma 能力的迁移路径，再评估 pgvector 是否接管。 |

默认 TTL 为首轮候选配置，按命中率、磁盘和复算成本调整。鉴权失败、超限、资源竞争和模型异常只记阶段尝试；失败风暴用退避和熔断处理。追问和流式中间文本受具体上下文影响，首版不做跨任务复用。

**页面控制**由工作区默认、批次或单任务覆盖构成。每层提供读、写两个独立开关，两者关闭即强制计算且禁止跨任务复用登记。新任务保存策略快照，进行中的任务保留原策略；页面另提供从某层重新执行，按勾选对该层和依赖它的后续层跳过缓存读取。关缓存或清缓存时仍照常记录该任务的阶段结果和报告。缓存中心可按来源、阶段、命中、模型、体积、创建和到期时间筛选，查看可读预览、复用来源及依赖链，可失效单条或清理指定层。管理员操作写审计日志。

## 4. 多任务、队列和历史页面

### 任务流和 RabbitMQ

1. 批量表单支持多文件、每行一个 URL、或混合提交。每项经限额、类型和参数校验后入库，接受项返回 `202`、批次 ID 和任务 ID，拒绝项返回独立原因。上传流限大小和并发，幂等键覆盖同次表单重复提交，原有单任务 API 调用同一服务层。
2. 实施阶段从 `~/infra/data` 启动现有 `postgres` 容器及 `mq` profile 的 `rabbitmq` 容器。可用 `docker compose start postgres` 和 `docker compose --profile mq start rabbitmq`，核对健康和持久卷；为 RabbitMQ 配置 `/meeting-agent` vhost 和专用发布、消费账户。端口保持回环 `5672`、管理端口 `15672`。验收后将 RabbitMQ 启动纳入主机启动运维。Python 使用 aio-pika，新增独立 Web、dispatcher、worker 用户级服务。
3. 用持久 direct exchange `meeting.tasks` 和持久 classic 队列 `meeting.io`、`meeting.cpu`、`meeting.cloud`、`meeting.gpu`。设 `meeting.dlx` 和诊断队列 `meeting.dead`。协议错误走死信路由，耗尽重试的业务失败由事务 outbox 派发诊断事件。单节点持久卷配合重建演练满足首版；跨节点高可用时再评估 quorum 队列。消息仅含 `dispatch_id, job_id, stage_run_id, schema_version, trace_id`，音频、转写、提示词、凭据由授权 worker 从业务存储取得。
4. 提交任务或阶段完成时，同事务写状态、事件和 outbox；dispatcher 用 `FOR UPDATE SKIP LOCKED` 批量认领并设置短租约，提交认领后才联网发布。使用持久消息、publisher confirm 和 mandatory，收到有效确认后才标记已发布。不可路由或 broker 中断时保留待派发记录，恢复后扫描。确认后进程崩溃会产生重复投递，由消费者幂等处理。
5. Worker 按 SQL 状态和租约认领阶段，维护心跳，持有者 token 对状态写入做围栏。新尝试只接管过期租约，旧 worker 的迟到结果不能覆盖新尝试。结果、阶段状态和下一阶段 outbox 在同事务提交，成功提交后手动 ACK。重复收到已终结的阶段可直接 ACK；收到仍被持有的阶段需走已持久化的续跑或过期扫描，避免热重投。GPU worker 的 prefetch 初始为 1，CPU 初始 2、云端初始 2–4，按限额和测试校准。长队列阶段设置 RabbitMQ delivery timeout 与 worker 截止时间；远程 ASR 提交后保存 provider job ID，后续轮询拆成延迟派发，等待时释放 worker。
6. 网络故障、429、5xx 采用带随机扰动的延迟重试，例如 10 秒、1 分钟、5 分钟。用 `outbox.available_at` 调度，遵守 Retry-After。认证失败、无效文件、无效模型等分类反馈。耗尽重试则终结并入死信，用户可以从最后可靠阶段重跑。外部 API 缺幂等键时，提交成功但本地落库前进程中断仍有重复计费可能；先按业务阶段流水重查 provider job ID，无法确认时明确标记，避免无限自动提交。取消等待阶段可立即生效，执行中的取消在结点、分片或供应商支持的边界生效。
7. RabbitMQ 只负责投递，PostgreSQL 为业务状态依据。设修复任务扫描长期未派发、心跳过期、文件丢失和 broker 重建后的未完成阶段。SSE 根据 `Last-Event-ID` 从 `job_events` 回放，恢复连接时仍可用状态 API 查最新结果。启动脚本增加 PostgreSQL、RabbitMQ、dispatcher、worker 的就绪检查及各自日志入口。队列暂不可用时，已正常入库的任务显示待派发。

### 页面交付

- **批量提交与任务队列**显示各输入、排队或运行状态、当前阶段、等待资源、可取消项、失败原因及重跑入口。区分待派发、已排队、执行中、等待 GPU、退避、成功、失败、取消。调度顺序会受资源和重试影响，队列位置显示估算。
- **历史记录**提供标题、类别、来源、日期、状态、模型、缓存命中筛选及游标分页。详情并排展示音视频元数据、带时间轴的转写和说话人、分类版本、面试题与评分或会议决策/行动项、运行耗时和费用估算、报告下载、阶段日志摘要。查看历史可以切换同一来源的版本或重跑记录。
- **可视化**先交付任务表格、详情的阶段耗时条、音频时间轴和状态统计。仪表盘按日期展示任务量、成功率、等待时间、模型耗时、缓存命中及异常；从图表可跳到已筛选的任务列表。
- **缓存与运维**交付分层开关、容量、命中率、搜索、失效、重跑以及 GPU 等待原因。跨工作区内容权限先于历史、事件、搜索、播放和缓存读取。首版继续通过 SSH 转发供 Mac 访问。

## 5. 单卡 GPU 调度与跨项目共用

**初期 GPU 并发一份租约。** Ollama、小型本地 ASR、其他模型服务统一走同一宿主机调度器。云端 ASR/LLM、普通 FFmpeg 及非 GPU 工作保持独立并行。可先用 Meeting Agent 消费 GPU 队列，调度入口设计成独立的宿主机用户服务和有权限的 Unix socket，其他项目可通过包装器申请、续租和释放。宿主机共享表建议建在独立数据库 `gpu_ops` 的 `scheduler` schema，包含 `gpu_requests`、`gpu_leases`、`gpu_profiles` 和 `gpu_events`；业务库只保存请求 ID 和阶段状态。以 GPU UUID 做唯一在租约束，申请/续租走原子事务，重启后通过活跃租约和实际进程复核。调度或采样不可用时，GPU 请求保持等待。

| 决策项 | 执行方式 |
| --- | --- |
| 用量观测 | NVML 为主，`nvidia-smi` 为诊断手段，采集 UUID、显存总量/实际空闲、利用率、温度、计算及图形进程，结合进程或 cgroup 判断项目；调用 Ollama `/api/ps` 获得驻留模型和显存。每 5 秒采样，调度前连续 3 次有效读数、实际启动前再查一次。PID 无法归属时按外部占用。 |
| 模型预算 | 对每种模型、量化、上下文、并发、批大小冷启动和热启动压测，登记显存总峰值和已驻留时的额外峰值。准入时实际空闲显存需 ≥ 预计额外峰值 + 安全余量，初值 1024 MiB，按实测波动上调。冷启动以模型总峰值作为额外需求。模型预算缺失时进独立低峰校准流程。 |
| 外部占卡 | 例如现有 `rag_law`。可归属、占用稳定且预算满足时允许一次准入；显存不足时标记 `waiting_gpu_external`，记录占用和差额。未经托管的进程可能在任务启动后突然加量，监控无法提供硬隔离；需保证显存上限时，安排独占窗口或将外部服务接入同一入口。外部进程由所属项目管理。 |
| 排队及公平 | 一张卡按可运行条件从等待队列选，项目间轮转、项目内以入队时间为主，等待超 5 分钟逐步提权。小任务可利用空隙，但老任务连续被跳过 3 次或等待超 10 分钟后为其预留下一次满足条件的时隙。GPU worker 将等待原因及请求 ID 持久化后释放 AMQP 消费槽，调度器准入后，由 dispatcher 按幂等键派发一次执行消息。租约受领超时则回收并重新排队。 |
| 运行中保护 | 初始心跳 10 秒、租约 45 秒，阶段有独立截止时间。等待超时按策略可继续等、手动取消或退到已授权的 CPU/云端线路，页面明确展示。显存变紧时停止新准入；出现 OOM 时只回收本项目持有的受管进程，检查显存实际释放后再以更小上下文或 batch 重试一次，失败则停止并保留阶段结果。心跳超时后先核对进程退出和显存，再允许下一份租约。 |
| Ollama | 受管窗口初始 `OLLAMA_MAX_LOADED_MODELS=1`、`OLLAMA_NUM_PARALLEL=1`，每模型限制 `num_ctx` 和输出长度。相同模型且有连续任务时可保留 30–60 秒热缓存；异模型等待或交还资源时，对自己管理的模型请求 `keep_alive=0` 并检查 NVML 和 `/api/ps`。已有其他 Ollama 调用统一接入口前，按外部负载计入。 |
| 监控和告警 | 页面实时显示显存、当前租约、等待原因和时长。Prometheus/Grafana 记录可用显存、受管队列深度、等待 p95、最近 OOM、租约失效、外部阻塞和监测中断。外部占用持续使最老 GPU 请求等待 >5 分钟发告警，>15 分钟升级；受管 OOM、租约超时且进程仍在、采样中断 >30 秒告警。按 GPU 和原因去重，恢复后关闭。接入外部通知渠道前另行配置；短时占用只在页面提示。 |

## 6. 实施顺序和验收

每阶段先用隔离库、RabbitMQ vhost 和合成样本验证，再逐步启用。保留旧输出和可读下载，功能以配置逐步切换。迁移演练保存备份和记录数量，回退可停新 worker、切回旧入口并保留新库供核对，避免删除新历史。数字门槛为首轮验收目标。

| 阶段 | 实施步骤 | 完成标准与验证方法 |
| --- | --- | --- |
| **P0 先修基础问题** | 建立阶段错误码。把分类、结构化输出校验、面试 `question.md` 和 `analyze.md` 的处理收敛到一套复用入口，校验批次题号和实际模型。加上传限额和流式落盘、URL 请求逐跳拦截回环/内网/保留 IP、清理静态文件直出及敏感日志。补最小工作区身份和权限。 | Python unittest 覆盖同名异内容、长录音、截断 JSON、模型回退、重复编号、文件名穿越、URL 重定向与 DNS 重绑定、跨工作区下载和日志脱敏。现有页面及单任务 API 回归通过。 |
| **P1 PostgreSQL、版本和历史** | 启动已有数据库，建库和迁移。为 Web、CLI 接同一存储接口，先写新记录与旧导出，再通过历史页读库。旧 `uploads/`、`output/asr/`、`output/summary/` 用 dry-run 导入，报告、题单和转写能可靠关联才回填同一任务；记录冲突、缺失和来源。 | 同名异内容分成两个媒体版本；同一 URL 改动产生新版本；缺原始时间保留为空。重启 Web 后历史、题单、转写、下载仍可读。重复导入不会新增重复记录。用独立目标恢复数据库及文件并校验样本哈希。 |
| **P2 分层缓存与界面** | 分阶段接入缓存索引、预览、策略版本、清理与下游键失效。先开 L2/L4/L6，再开文件缓存和分片缓存，逐层测收益。旧同名缓存仅供导入和人工复核。 | 每层做两次相同输入，观察命中及外部调用次数；修改模型/提示词/上游内容后对应层失效；切换四种读写组合符合预期。清缓存后历史文件可下载。并发同键只发布一个有效产物；构建进程被终止后租约可接管。 |
| **P3 RabbitMQ、批量与可恢复进度** | 增加提交批次、outbox、阶段 worker、DLQ 和监控。用数据库事件替代进程内状态及 SSE 队列。保留现有单任务 REST 契约，再切到新执行方式。 | 一次提交至少 3 个文件和 3 个 URL，含一项无效输入。注入 broker 停机、发布确认后退出、消费提交后退出、provider 429、worker 重启和取消；核对无任务丢失、无重复业务记录、SSE 可续传、终态可查询、失败可续跑。持续负载下按 CPU/云端预算限制并发。 |
| **P4 GPU 和告警** | 先仅采样和试算准入。校准一个小型模型，再接入 Ollama 和一个本地 ASR worker。建立 `rag_law` 和其他项目的受管接入约定。 | 用可控负载模拟约 2.4 GiB 外占、显存不足、Ollama 驻留、受管 OOM、租约超时、两项目争抢。确认 CPU/云端仍继续、每张卡最多一个受管租约、外部任务不被中断、等待不过早放行、恢复后能排空，告警与恢复可观察。以 24 小时混合负载观察误报率和 OOM，再启用默认 GPU 调度。 |
| **P5 运维与优化** | 加成本/延迟看板、上传及数据库保留策略、任务审计和下载权限、真实备份恢复定期演练。对话和追问按工作区持久化，随后评估向量检索和关键词检索效果。 | 用脱敏样本回归转写、各类报告、追问和删除。验证删除权限、数据库连接池、列表查询及监控指标。定期记录依赖、模型和提示词升级的前后效果。 |

上线验收应同时覆盖短录音、长面试、视频文件、字幕 URL、无语音、不可访问 URL、两个同名文件及同一 URL 的两个内容版本。耗时较长或含个人内容的样本优先在隔离测试中使用合成或脱敏数据，另对已授权的数据校验报告质量。

## 参考

- PostgreSQL [JSONB 和索引](https://www.postgresql.org/docs/17/datatype-json.html)、[SELECT 行锁及 SKIP LOCKED](https://www.postgresql.org/docs/17/sql-select.html)。
- RabbitMQ [发布确认与消费确认](https://www.rabbitmq.com/docs/confirms)、[Quorum Queues](https://www.rabbitmq.com/docs/quorum-queues)。
- NVIDIA [NVML 设备查询](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)。Ollama [并发、驻留和上下文配置](https://docs.ollama.com/faq)、[驻留模型查询](https://docs.ollama.com/api/ps)。
