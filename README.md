# 会议助手智能体 (Meeting Agent)

基于 ASR + LLM 的音视频内容分析智能体。支持会议录音、面试录音、通用录音和在线视频输入，自动输出结构化的会议纪要、面试评估报告或内容总结。

## 功能特性

- **多类型录音自动分类** — 自动识别会议 / 面试 / 通用录音，路由到对应分析策略
- **视频总结** — 支持本地视频文件和在线视频链接（YouTube、Bilibili 等 1000+ 站点）
- **ASR 语音转写** — 基于阿里云百炼 Paraformer-v2，支持说话人分离和时间戳
- **LLM 智能分析** — 基于通义千问系列模型，输出结构化 JSON 并格式化为 Markdown
- **长文本滑动窗口** — 分段分析时相邻片段保留 3 句重叠，避免切割边界上下文丢失
- **多级缓存** — ASR 结果和 Summary 结果均支持缓存，重复处理同一文件时秒级完成
- **耗时追踪** — 记录 ASR 耗时、LLM 耗时、总耗时，写入总结文档

## 处理流水线

```
音频/视频输入
    │
    ├─ 视频 → yt-dlp 下载音频 / ffmpeg 提取音频
    │
    ├─ 音频预处理 → 单声道 16kHz WAV
    │
    ├─ ASR 转写 (Paraformer-v2) → 带时间戳和说话人标签的句子列表
    │
    ├─ 文本清洗 → 去语气词、合并短句、清理空白
    │
    ├─ 录音类型分类 (LLM) → meeting / interview / general
    │
    └─ LLM 分析 → 结构化 JSON → Markdown 输出
```

## 技术栈

| 组件 | 技术 |
|------|------|
| ASR | 阿里云百炼 Paraformer-v2 |
| LLM | 通义千问 (qwen-plus / qwen3 系列) |
| 视频下载 | yt-dlp (Python 模块) |
| 音频处理 | ffmpeg + pydub |
| HTTP | httpx (async) |
| CLI | Typer + Rich |
| 配置 | pydantic-settings + .env |

## 安装

### 1. 克隆项目

```bash
git clone <repo-url>
cd meeting_agent
```

### 2. 创建 conda 环境

```bash
conda create -n meeting_agent python=3.12 -y
conda activate meeting_agent
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 4. 安装 ffmpeg (视频功能需要)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 5. 配置 API Key

创建 `.env` 文件：

```bash
cp .env.example .env   # 或手动创建
```

编辑 `.env`，填入阿里云百炼 API Key：

```env
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

> API Key 获取地址：https://bailian.console.aliyun.com/
>
> ASR 和 LLM 共用同一个 Key。如需单独配置 LLM Key，可额外设置 `LLM_API_KEY` 和 `LLM_BASE_URL`。

### 6. 配置 Bilibili cookies (可选)

下载 Bilibili 视频时，需要导出浏览器 cookies 以避免 412 反爬错误：

1. 在 Chrome 中安装扩展 **"Get cookies.txt LOCALLY"**
2. 打开 bilibili.com 并确保已登录
3. 点击扩展图标 → 导出 cookies（Netscape 格式）
4. 将文件保存到 `data/cookies/www.bilibili.com_cookies.txt`

## 使用方式

### 分析音频 (自动识别类型)

```bash
# 分析本地音频文件
python main.py analyze /path/to/recording.wav

# 分析远程音频 URL
python main.py analyze "https://example.com/audio.mp3"

# 输出 JSON 格式
python main.py analyze recording.wav --output json

# 显示详细日志
python main.py analyze recording.wav -v
```

支持的音频格式：wav / mp3 / m4a / flac / ogg 等。

### 分析视频

```bash
# 分析本地视频
python main.py video /path/to/video.mp4

# 分析 Bilibili 视频
python main.py video "https://www.bilibili.com/video/BV1xxxxxx/"

# 分析 YouTube 视频
python main.py video "https://www.youtube.com/watch?v=xxxxxxx"
```

支持 YouTube、Bilibili 等 1000+ 视频站点。视频总结结果以视频标题命名。

### 仅 ASR 转写 (调试用)

```bash
python main.py transcript-only /path/to/audio.wav
```

## 输出目录结构

```
output/
├── asr/                                    # ASR 转写缓存
│   └── {name}_{asr_model}.txt
│
└── summary/                                # 总结文档
    ├── meeting/                            # 会议纪要
    │   └── {name}_{asr}_{llm}.md
    ├── interview/                          # 面试分析
    │   └── {name}_{asr}_{llm}/
    │       ├── question.md                 # 面试问题列表
    │       └── analyze.md                  # 逐题分析报告
    ├── video/                              # 视频总结
    │   └── {title}_{asr}_{llm}.md
    └── other/                              # 其他录音
        └── {name}_{asr}_{llm}.md
```

## 配置项

所有配置通过 `.env` 文件或环境变量设置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DASHSCOPE_API_KEY` | (必填) | 阿里云百炼 API Key |
| `LLM_API_KEY` | 同 DASHSCOPE_API_KEY | LLM 专用 API Key |
| `LLM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | LLM API 地址 |
| `LLM_MODEL` | `qwen-plus` | LLM 模型名称 |
| `ASR_MODEL` | `paraformer-v2` | ASR 模型名称 |
| `ASR_SAMPLE_RATE` | `16000` | 音频采样率 |
| `MAX_TOKENS_PER_CHUNK` | `6000` | 分段分析每片最大 token 数 |
| `MIN_SENTENCE_LENGTH` | `10` | 短句合并阈值 (字符数) |
| `OUTPUT_DIR` | `./output` | 输出目录 |

## 项目结构

```
meeting_agent/
├── main.py                 # CLI 入口 (analyze / video / transcript-only)
├── config.py               # 配置管理
├── requirements.txt
├── agent/
│   ├── asr_client.py       # DashScope Paraformer ASR 客户端
│   ├── llm_client.py       # LLM 客户端 (OpenAI 兼容接口)
│   ├── pipeline.py         # 处理流水线编排
│   ├── prompts.py          # Prompt 模板
│   ├── classifier.py       # 录音类型分类
│   ├── text_processor.py   # 文本清洗与分段 (含滑动窗口)
│   ├── video_client.py     # 视频处理 (yt-dlp + ffmpeg)
│   └── cache.py            # 缓存管理
├── models/
│   └── schemas.py          # 数据模型 (Sentence, Transcript, MeetingResult...)
├── utils/
│   └── audio.py            # 音频预处理工具
└── data/
    └── cookies/            # 浏览器 cookies (Bilibili 等)
```

## License

MIT
