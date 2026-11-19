# 会议历史问答：交付与运维

## 版本与入口

本轮在 `agentic` 分支按 K0 → K5 实施；不修改既有 `zane` 分支。代码提交与发布标签：

| 阶段 | 交付 | 提交 / 标签 |
| --- | --- | --- |
| K0 | 80 个冻结问题、覆盖审计、无证据不生成私有事实 | `41e28ad` |
| K1 | 会议事件、人工确认、证据与增量回填 | `bbb2657` |
| K2 | 日历解析、模板 SQL、按事件去重、名单和主题问答 | `48d7f53` / `knowledge-baseline-v1` |
| K3 | 授权事件过滤、词法/向量 RRF、可选精排、多轮与版本缓存 | `de2f628` / `knowledge-hybrid-v1` |
| K4 | 可撤销人物别名、时态关系和最多三跳、受约束统计 AST | `70e4a26` / `knowledge-query-target-v1` |
| K5 | 主页面入口、旧聊天兼容、覆盖/反馈/观测、恢复与上线 | `knowledge-release-rc2`（功能验收版本） |

Mac 页面：<http://127.0.0.1:19190/knowledge.html>。首页右上角「历史问答」也可进入。需要保持原有 SSH 转发。

## 使用

1. 完成分析后，知识 worker 通常在 10 秒内读取已有报告和转写，不会重新 ASR 或分析。
2. 在「历史问答」下方查看事件、候选日期、人员与证据；确认发生日期及真实参会名单。
3. 提问「上周的会议有哪些人参加？」「去年今天的会议讨论了什么主题？」。默认类别为会议；查询面试时在问题中写“面试”。
4. 答案显示范围、完整性和来源链接；同一浏览器标签页刷新后保留会话，可追问“其中张三参加几次？”。

只有日期缺年份时保持未知；文件创建/上传时间不能代替会议日期。Speaker 标签属于单份转写，不能自动当作真实姓名。报告列出的姓名属于待确认信息。只有日期已确认的记录进入精确日期统计，覆盖不足会明确提示。

「最近七天」是从当前时刻向前 168 小时；「上周」是本地自然周。默认按开始日归属；写“2026年11月18日期间举行过的会议”使用区间重叠。准确时间可经 `PUT /api/knowledge/sessions/{id}/confirmation` 提交带时区的 `event_start_at`、`event_end_at`。只有日期的记录不混入精确时刻/重叠统计。日期与准确开始时间冲突会拒绝保存。

## 存储与边界

沿用 PostgreSQL **meeting_agent** 数据库、**meeting** schema，迁移 `004_knowledge` 增加：

- 事实：`meeting_sessions`、`meeting_recordings`、`meeting_time_candidates`、`people`、`person_aliases`、`speaker_links`、`meeting_participants`、`meeting_topics`。
- 证据与关系：`evidence_spans`、`knowledge_entities`、`knowledge_relations`。
- 投影、版本和审计：`knowledge_chunks`、`knowledge_builds`、`knowledge_state`、`query_runs`、`query_feedback`。

会议内容哈希相同才自动归并；不同内容但属于同场会议的人工合并尚未提供。人工日期与名单在重建时保留。关系通过携带证据的 API 登记，不从“提到某人”推断参会。别名可撤销，跨工作区读取不允许。当前沿用个人工作区与现有访问控制，不是多租户账号产品。

模板答案在 PostgreSQL `query_runs` 中复用 60 秒，键包含范围、工作区、事实版本和模型配置。纠错、别名变化和软删除提升版本；查询使用同一数据库快照。搜索不复用最终结果；向量由 `knowledge_chunks` 保存，携带模型与来源版本，过期投影不参与检索。

向量使用 SQL 授权过滤后的 JSONB 精确余弦参考实现，最多扫描 20,000 段、返回 20 段；截断与水位显式返回。Neo4j 和 pgvector ANN 未成为运行依赖。百万段检索需要经过容量验证后采用分区/ANN，不能将小样本验收当作容量结论。

## 开关与服务

```dotenv
KNOWLEDGE_ENABLED=true
KNOWLEDGE_SEMANTIC_ENABLED=false
KNOWLEDGE_RERANK_ENABLED=false
KNOWLEDGE_AGGREGATE_ENABLED=false
```

默认上线日历/名单/主题模板、词法检索及关系 API。按工作区和事件范围过滤后才进行模型检索；启用语义/精排需验证配置的模型与配额，会向对应模型供应商发送查询或转写片段。失败降级到词法结果并显示原因。

`/api/knowledge/aggregate` 是实验接口：模型只输出严格白名单 AST（确认参会次数、按人物分组、Top 1–20），服务器编译 SQL；拒绝任意 SQL、工作区字段及额外字段，查询超时 2 秒。它不是通用自由 SQL Agent。开放式跨项目统计还需要扩展白名单与独立评测，默认保持关闭。

安装和检查：

```bash
uv sync --locked
# 在私有环境注入 MIGRATION_DATABASE_URL，再执行迁移；不得使用 Web 运行账号建表。
uv run --locked alembic upgrade head
install -m 644 deploy/meeting-agent-knowledge.service ~/.config/systemd/user/
install -m 644 deploy/meeting-agent-knowledge-observe.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now meeting-agent-knowledge.service
./start.sh restart
systemctl --user start meeting-agent-knowledge-observe.service
systemctl --user status meeting-agent.service meeting-agent-knowledge.service
journalctl --user -u meeting-agent-knowledge.service -n 30 --no-pager
```

迁移账号给运行账号授予新表 DML 和序列权限，保留既有默认授权。原 RabbitMQ dispatcher、io/cpu/cloud/gpu workers 与 GPU watch 继续运行。本次知识回填使用有水位的 SQL 扫描，不新增 RabbitMQ 队列，不占 GPU。

## 验证、恢复与回滚

在隔离数据库和 RabbitMQ vhost 下执行：

```bash
# TEST_DATABASE_URL / TEST_AMQP_URL / TEST_GPU_DATABASE_URL 由私有验收环境注入。
uv run --locked python -m unittest discover -s tests -v
node tests/question_download_frontend.cjs
```

本轮隔离回归：56 个测试用例，54 个通过、2 个跳过；跳过的 Node 下载回归另在 Mac 执行通过，真实 GPU 负载测试本轮未重跑。80 条问题分别做解析和实际 SQL 结果集合验收，另测闰日、跨时区、DST、精确滚动时间、跨午夜、跨工作区、纠错缓存失效、两跳关系、危险 AST 拒绝和失败回填隔离。该结果不等于真实语料的开放意图正确率或检索 Recall@20 已达标。

覆盖：`GET /api/knowledge/coverage`；查询计数：`GET /api/knowledge/metrics`；页面可提交有帮助/需改进反馈。失败回填只保存错误类型，5 分钟后自动重试，每批最多 100 条；单条失败不会阻塞其他报告。

恢复步骤：停止知识 worker，运行 `uv run --locked python -m knowledge.worker --rebuild --once`，再启动 worker；后台继续处理剩余批次。此操作保留人工确认与软删除，不调用 ASR。回滚开关 `KNOWLEDGE_ENABLED=false` 可停用新入口，关闭语义/精排/统计实验可退回模板通路。先回滚应用源码和配置，保留新增表；不要直接降级删除证据表。

上线前备份数据库和源码到部署账号私有目录，保留 `.env`。24 小时观测服务每分钟检查健康与覆盖，原子写入 `data/knowledge-observation.json`（权限 0600）。它只做真实被动观测，不制造用户流量；`status=completed` 也不代表质量/混合负载自动达标。当前发布为 RC，24 小时观测、真实语料人物 precision/recall、旧路径对照 Recall@20、开放意图执行率及 1 万事件/100 万段 P95 仍需独立验收，不标记为已通过。


## 本次 Ubuntu 上线记录（2026-11-20）

- 数据库备份：`~/.config/meeting-agent/backups/knowledge-c9996bf/meeting_agent.dump`；源码、配置备份位于同目录，仅部署账号可读。
- 增量迁移已完成，新增表运行权限已验证；Web、dispatcher、四类 worker、GPU watch、knowledge worker 和观测服务运行正常。
- 已有 9 条分析回填成功、0 条失败，生成 9 场事件与 522 段转写投影；0 场确认日期、0 段云端向量（默认关闭语义）。当前没有 meeting 类历史任务，因此示例会议问题返回无匹配；面试查询显示待确认日期覆盖，不伪造答案。
- 上线冒烟验证首页、问答、历史、批量页面；日期/名单/主题查询与缺年份澄清。指标接口发现 JSONB 分组参数错误，已增加真实 PostgreSQL API 回归并修正为分组别名，补丁发布为 RC2。
- Mac 的 `19190` SSH 转发已核实可用，无需改动。24 小时被动观测正在运行，尚未完成长期验收。
