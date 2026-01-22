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
    port: int = 8003
    workers: int = 1


class WhisperXConfig(BaseModel):
    """WhisperX 配置"""
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "int8"  # float16, int16, int8, int8_float16
    batch_size: int = 24  # 24GB显卡推荐 24-32，太高会 OOM
    language: str | None = None
    # 对齐模型配置：{language_code: model_name}
    # 例如 {"zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"}
    # 如果某个语言的对齐模型加载失败，可以在这里指定替代模型
    align_models: dict[str, str] | None = None


class DiarizationConfig(BaseModel):
    """说话人分离配置"""
    enabled: bool = True
    huggingface_token: str | None = Field(default_factory=lambda: os.getenv("HF_TOKEN"))
    min_speakers: int | None = None
    max_speakers: int | None = None


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
    output: OutputConfig = Field(default_factory=OutputConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


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

