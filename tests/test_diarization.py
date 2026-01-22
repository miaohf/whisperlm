"""说话人分离测试"""

import pytest

from whisperlm.config import get_settings
from whisperlm.services.whisperx_service import WhisperXService


def test_whisperx_service_init():
    """测试 WhisperX 服务初始化"""
    service = WhisperXService()
    assert service is not None
    assert not service.is_loaded


def test_whisperx_config():
    """测试 WhisperX 配置"""
    settings = get_settings()
    assert settings.whisperx.model == "large-v3"
    assert settings.whisperx.compute_type == "float16"
    assert settings.whisperx.batch_size == 16


def test_diarization_config():
    """测试说话人分离配置"""
    settings = get_settings()
    assert settings.diarization.enabled is True


def test_gpu_info():
    """测试 GPU 信息获取"""
    service = WhisperXService()
    gpu_info = service.get_gpu_info()
    assert "available" in gpu_info
    assert isinstance(gpu_info["available"], bool)

