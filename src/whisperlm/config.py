"""配置管理模块"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseModel):
    """服务配置"""
    host: str = "0.0.0.0"
    port: int = 8001
    workers: int = 1


class WhisperXConfig(BaseModel):
    """WhisperX 配置"""
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"  # float16, int16, int8, int8_float16
    batch_size: int = 16
    language: str | None = None
    # 对齐模型配置：{language_code: model_name}
    # 例如 {"zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"}
    align_models: dict[str, str] | None = None


class DiarizationConfig(BaseModel):
    """说话人分离配置"""
    enabled: bool = True
    # 优先级：config.yaml > .env 文件 > 环境变量 HF_TOKEN
    huggingface_token: str | None = Field(
        default=None,
        description="Hugging Face Token，用于访问 pyannote 模型。可从环境变量 HF_TOKEN 或 .env 文件读取"
    )
    min_speakers: int | None = None
    max_speakers: int | None = None

    def __init__(self, **data):
        # 如果配置文件中没有提供 token 或值为空字符串，尝试从环境变量读取
        huggingface_token = data.get("huggingface_token")
        if not huggingface_token or huggingface_token.strip() == "":
            env_token = os.getenv("HF_TOKEN")
            if env_token:
                data["huggingface_token"] = env_token
        super().__init__(**data)


class LLMFeaturesConfig(BaseModel):
    """LLM 功能开关"""
    semantic_segmentation: bool = True    # 按语义边界智能断句/合并
    error_correction: bool = True         # 修复 ASR 识别错误
    expression_optimization: bool = True  # 优化口语表达


class LLMConfig(BaseModel):
    """LLM 配置"""
    enabled: bool = False
    provider: str = "vllm"               # vllm, openai, ollama, azure, anthropic
    model: str = "Qwen/Qwen3-32B"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    timeout: int = 120
    max_retries: int = 3
    features: LLMFeaturesConfig = Field(default_factory=LLMFeaturesConfig)


class TranslationConfig(BaseModel):
    """翻译配置"""
    enabled: bool = False
    target_language: str = "zh"
    style: str = "natural"               # natural, formal, casual


class OutputConfig(BaseModel):
    """输出配置"""
    formats: list[str] = Field(default_factory=lambda: ["json", "srt", "vtt"])
    include_word_timestamps: bool = True
    include_confidence: bool = True


class Settings(BaseSettings):
    """应用配置"""
    server: ServerConfig = Field(default_factory=ServerConfig)
    whisperx: WhisperXConfig = Field(default_factory=WhisperXConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        case_sensitive = False


def _expand_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """递归展开配置中的环境变量"""
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _expand_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            result[key] = os.getenv(env_var, "")
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path | None = None) -> Settings:
    """加载配置文件"""
    if config_path is None:
        possible_paths = [
            Path("config.yaml"),
            Path("config.yml"),
            Path(__file__).parent.parent.parent.parent / "config.yaml",
        ]
        for p in possible_paths:
            if p.exists():
                config_path = p
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
        config_dict = _expand_env_vars(config_dict)
        return Settings(**config_dict)

    return Settings()


_settings: Settings | None = None


def get_settings() -> Settings:
    """获取全局配置实例"""
    global _settings
    if _settings is None:
        _settings = load_config()
    return _settings

