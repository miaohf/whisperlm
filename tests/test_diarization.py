"""WhisperX 服务与说话人分离配置测试"""

import pytest

from whisperlm.config import get_settings
from whisperlm.services.whisperx_service import WhisperXService


def test_whisperx_service_init():
    """WhisperX 服务初始化后未加载模型"""
    service = WhisperXService()
    assert service is not None
    assert not service.is_loaded


def test_whisperx_config():
    """WhisperX 默认配置符合预期"""
    settings = get_settings()
    assert settings.whisperx.model == "large-v3"
    assert settings.whisperx.compute_type in ("float16", "int8", "int8_float16", "int16")
    assert settings.whisperx.batch_size > 0


def test_diarization_config():
    """说话人分离默认启用"""
    settings = get_settings()
    assert settings.diarization.enabled is True


def test_gpu_info():
    """GPU 信息结构正确"""
    service = WhisperXService()
    gpu_info = service.get_gpu_info()
    assert "available" in gpu_info
    assert isinstance(gpu_info["available"], bool)
