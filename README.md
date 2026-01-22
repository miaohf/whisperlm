# WhisperLM

一个结合 WhisperX 精确转录和 LLM 语义优化的智能语音转文字服务。

## ✨ 特性

- 🎯 **精确转录**：基于 WhisperX 的词级时间戳对齐（精度 ~50ms）
- 👥 **说话人分离**：自动识别和标注不同说话人（基于 pyannote）
- 🧠 **语义优化**：LLM 智能断句、修复 ASR 错误、优化表达
- 🌍 **多语言翻译**：支持 100+ 语言的高质量翻译
- 🚀 **高性能**：支持 GPU 加速，批量处理
- 📡 **RESTful API**：易于集成的 HTTP 接口
- 🔄 **一体化处理**：转录、对齐、说话人分离一次完成

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         WhisperLM                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  音频    │───►│ WhisperX │───►│   LLM    │───►│  输出    │  │
│  │  输入    │    │  Pipeline│    │  优化器  │    │  字幕    │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                       │                │                        │
│                       ▼                ▼                        │
│                 ┌──────────┐    ┌──────────┐                   │
│                 │ Whisper  │    │ 语义分段 │                   │
│                 │ wav2vec2 │    │ 错误修复 │                   │
│                 │ pyannote │    │ 翻译优化 │                   │
│                 └──────────┘    └──────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 处理流程

```
1. 音频输入
      │
      ▼
2. VAD 预处理（过滤静音段）
      │
      ▼
3. Whisper 转录（生成初始文本）
      │
      ▼
4. wav2vec2 词级对齐（精确时间戳）
      │
      ▼
5. pyannote 说话人分离（标注说话人）
      │
      ▼
6. LLM 语义处理（可选）
   ├── 智能断句（按语义边界）
   ├── ASR 错误修复
   ├── 表达优化
   └── 多语言翻译
      │
      ▼
7. 输出字幕（JSON/SRT/VTT）
```

## 🚀 快速开始

### 环境要求

- Python >= 3.10
- CUDA >= 11.8（推荐，支持 CPU 但较慢）
- FFmpeg >= 4.0
- 8GB+ GPU 显存（推荐 16GB+）

### 安装

```bash
# 克隆项目
git clone https://github.com/your-org/whisperlm.git
cd whisperlm

# 使用 uv 安装依赖（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 配置

复制配置文件并修改：

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`：

```yaml
# 服务配置
server:
  host: "0.0.0.0"
  port: 8001
  workers: 1

# WhisperX 配置（推荐：float16）
whisperx:
  model: "large-v3"           # tiny, base, small, medium, large-v2, large-v3
  device: "cuda"              # cuda, cpu
  compute_type: "float16"     # float16 精度更高，显存占用 ~8GB
  batch_size: 16
  language: null              # 自动检测，或指定如 "en", "zh"

# 说话人分离配置
diarization:
  enabled: true
  huggingface_token: "${HF_TOKEN}"  # 需要申请 pyannote 权限
  min_speakers: null          # 最小说话人数，null 为自动
  max_speakers: null          # 最大说话人数，null 为自动

# LLM 配置（使用本地 vLLM）
llm:
  enabled: true               # 是否启用 LLM 优化
  provider: "vllm"            # vllm, openai, ollama, azure, anthropic
  model: "Qwen/Qwen3-32B"     # 模型名称
  base_url: "http://localhost:8000/v1"  # vLLM 服务地址
  api_key: ""                 # vLLM 本地部署无需 API key
  
  # 功能开关
  features:
    semantic_segmentation: true   # 语义分段
    error_correction: true        # ASR 错误修复
    expression_optimization: true # 表达优化

# 翻译配置（可选）
translation:
  enabled: false
  target_language: "zh"           # 目标语言
  style: "natural"                # natural, formal, casual

# 输出配置
output:
  formats: ["json", "srt", "vtt"]
  include_word_timestamps: true
  include_confidence: true
```

### 环境变量

创建 `.env` 文件：

```bash
# Hugging Face Token（必需，用于 pyannote 说话人分离）
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

### 启动 vLLM 服务

```bash
# 启动 vLLM 服务（Qwen3-32B）
vllm serve Qwen/Qwen3-32B --port 8000 --tensor-parallel-size 1
```

### 启动服务

```bash
# 开发模式
uv run python -m whisperlm.main

# 生产模式
uv run gunicorn whisperlm.main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8001
```

## 📖 API 文档

### 核心接口：转录

**POST** `/api/v1/transcribe`

```bash
curl -X POST "http://localhost:8002/api/v1/transcribe" \
  -F "file=@audio.mp3" \
  -F "language=en" \
  -F "diarization=true" \
  -F "llm_optimize=true"
```

**请求参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| file | File | ✅ | - | 音频/视频文件 |
| language | string | ❌ | auto | 语言代码，auto 为自动检测 |
| diarization | bool | ❌ | true | 是否启用说话人分离 |
| llm_optimize | bool | ❌ | true | 是否启用 LLM 优化 |
| output_format | string | ❌ | json | 输出格式：json/srt/vtt |
| min_speakers | int | ❌ | null | 最小说话人数 |
| max_speakers | int | ❌ | null | 最大说话人数 |

**响应示例：**

```json
{
  "task_id": "abc123",
  "status": "completed",
  "language": "en",
  "duration": 125.4,
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.52,
      "text": "Welcome to today's discussion about artificial intelligence.",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Welcome", "start": 0.0, "end": 0.42, "confidence": 0.98},
        {"word": "to", "start": 0.44, "end": 0.52, "confidence": 0.99},
        {"word": "today's", "start": 0.54, "end": 0.92, "confidence": 0.97},
        {"word": "discussion", "start": 0.94, "end": 1.48, "confidence": 0.96},
        {"word": "about", "start": 1.50, "end": 1.72, "confidence": 0.99},
        {"word": "artificial", "start": 1.74, "end": 2.28, "confidence": 0.95},
        {"word": "intelligence.", "start": 2.30, "end": 3.12, "confidence": 0.94}
      ],
      "confidence": 0.97
    },
    {
      "id": 1,
      "start": 4.80,
      "end": 8.25,
      "text": "Thank you for having me. It's a pleasure to be here.",
      "speaker": "SPEAKER_01",
      "words": [...],
      "confidence": 0.96
    }
  ]
}
```

### 转录 + 翻译

**POST** `/api/v1/transcribe-translate`

```bash
curl -X POST "http://localhost:8001/api/v1/transcribe-translate" \
  -F "file=@video.mp4" \
  -F "source_language=en" \
  -F "target_language=zh" \
  -F "translation_style=natural"
```

**额外参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| target_language | string | ✅ | - | 目标语言代码 |
| translation_style | string | ❌ | natural | 翻译风格：natural/formal/casual |

**响应示例：**

```json
{
  "task_id": "def456",
  "status": "completed",
  "source_language": "en",
  "target_language": "zh",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.52,
      "text": "Welcome to today's discussion about artificial intelligence.",
      "translated_text": "欢迎来到今天关于人工智能的讨论。",
      "speaker": "SPEAKER_00",
      "confidence": 0.97
    }
  ]
}
```

### 兼容旧版接口

为了兼容旧版 STT 服务，WhisperLM 提供以下兼容接口：

**POST** `/transcribe/`

```bash
curl -X POST "http://localhost:8001/transcribe/" \
  -F "file=@audio.mp3"
```

**响应格式（兼容旧版）：**

```json
{
  "status": "success",
  "results": [
    {
      "start": 0.0,
      "end": 4.52,
      "text": "Welcome to today's discussion about artificial intelligence."
    }
  ]
}
```

> 注意：兼容接口内部使用 WhisperX 一体化处理，返回结果已包含说话人信息的精确对齐，但响应格式保持旧版兼容。

### 健康检查

**GET** `/health`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "whisperx": {
    "model": "large-v3",
    "device": "cuda",
    "loaded": true
  },
  "diarization": {
    "model": "pyannote/speaker-diarization-3.1",
    "loaded": true
  },
  "llm": {
    "provider": "vllm",
    "model": "Qwen/Qwen3-32B",
    "connected": true
  },
  "gpu": {
    "available": true,
    "name": "NVIDIA RTX 4090",
    "memory_total": "24GB",
    "memory_used": "8GB"
  }
}
```


## 📁 项目结构

```
whisperlm/
├── src/
│   └── whisperlm/
│       ├── __init__.py
│       ├── main.py                 # FastAPI 应用入口
│       ├── config.py               # 配置管理
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes.py           # API 路由
│       │   ├── legacy_routes.py    # 兼容旧版接口
│       │   └── models.py           # Pydantic 模型
│       ├── services/
│       │   ├── __init__.py
│       │   ├── whisperx_service.py # WhisperX 核心服务
│       │   ├── llm_service.py      # LLM 优化服务
│       │   └── task_service.py     # 异步任务管理
│       ├── processors/
│       │   ├── __init__.py
│       │   ├── audio_processor.py  # 音频预处理
│       │   └── subtitle_processor.py # 字幕格式转换
│       └── utils/
│           ├── __init__.py
│           ├── formats.py          # 格式转换工具
│           └── prompts.py          # LLM Prompts
├── tests/
│   ├── test_transcribe.py
│   ├── test_diarization.py
│   └── test_llm.py
├── config.example.yaml
├── pyproject.toml
└── README.md
```

## ⚙️ 高级配置

### LLM Prompt 自定义

编辑 `src/whisperlm/utils/prompts.py`：

```python
# 语义分段优化 Prompt
SEMANTIC_SEGMENTATION_PROMPT = """
你是一个专业的字幕编辑。请对以下转录文本进行语义分段优化。

规则：
1. 每段应该是一个完整的语义单元（一个完整的想法/观点）
2. 保持原有时间戳的准确性
3. 修复明显的 ASR 错误（如错别字、漏字）
4. 不要改变原意，保持口语化表达

输入：
{transcription}

请以 JSON 格式输出优化后的字幕。
"""

# 翻译优化 Prompt
TRANSLATION_PROMPT = """
你是一个专业的{target_language}翻译。请翻译以下字幕。

要求：
1. 保持口语化、自然流畅
2. 适当调整语序以符合{target_language}表达习惯
3. 保留专业术语的准确性
4. 控制每条字幕长度，适合阅读

原文：
{text}

翻译：
"""
```

### 性能调优

```yaml
# 推荐配置（8GB+ GPU）- 精度最高
whisperx:
  model: "large-v3"
  compute_type: "float16"     # float16，精度高，显存占用 ~8GB
  batch_size: 16

# 高性能配置（16GB+ GPU）
whisperx:
  model: "large-v3"
  compute_type: "float16"
  batch_size: 32              # 更大批次，速度更快

# 低显存配置（8GB GPU 显存紧张时）
whisperx:
  model: "large-v3"
  compute_type: "int8"        # int8 量化，显存占用 ~5GB
  batch_size: 16
```

### 使用本地 LLM（vLLM + Qwen3-32B）

```bash
# 启动 vLLM 服务
vllm serve Qwen/Qwen3-32B --port 8000

# 或使用更多 GPU 并行
vllm serve Qwen/Qwen3-32B --port 8000 --tensor-parallel-size 2
```

```yaml
# config.yaml 配置
llm:
  provider: "vllm"
  model: "Qwen/Qwen3-32B"
  base_url: "http://localhost:8001/v1"
  api_key: ""
```

## 🔧 常见问题

### 1. pyannote 权限申请

1. 访问 https://huggingface.co/pyannote/speaker-diarization-3.1
2. 点击 "Access repository" 申请权限
3. 获取 Hugging Face Token：https://huggingface.co/settings/tokens
4. 设置环境变量 `HF_TOKEN`

### 2. CUDA 内存不足

```yaml
# 降低模型大小和批次
whisperx:
  model: "medium"      # 使用小模型
  compute_type: "int8" # 使用 int8 量化
  batch_size: 4        # 减小批次大小
```

### 3. 转录速度慢

- 确保使用 GPU（`device: "cuda"`）
- 增大 `batch_size`
- 使用 `compute_type: "float16"` 而非 `float32`

### 4. 说话人分离不准确

```yaml
diarization:
  min_speakers: 2      # 明确指定说话人数量
  max_speakers: 2
```

### 5. LLM 超时

```yaml
llm:
  timeout: 120         # 增加超时时间
  max_retries: 3       # 增加重试次数
```

## 📊 性能基准

| 配置 | 10分钟音频处理时间 | GPU 显存占用 |
|------|-------------------|--------------|
| large-v3 + float16 | ~45s | ~8GB |
| large-v3 + int8 | ~60s | ~5GB |
| medium + float16 | ~25s | ~4GB |
| small + float16 | ~15s | ~2GB |
| small + CPU | ~180s | - |

*测试环境：NVIDIA RTX 4090, Intel i9-13900K*

## 🔗 相关项目

- [WhisperX](https://github.com/m-bain/whisperX) - 核心转录引擎
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - 高性能 Whisper 实现
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) - 说话人分离

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

