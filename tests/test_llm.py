"""LLM 服务测试"""

import pytest

from whisperlm.config import get_settings
from whisperlm.services.llm_service import LLMService
from whisperlm.api.models import SegmentResponse


def test_llm_service_init():
    """测试 LLM 服务初始化"""
    service = LLMService()
    assert service is not None


def test_llm_config():
    """测试 LLM 配置"""
    settings = get_settings()
    assert settings.llm.provider == "vllm"
    assert settings.llm.model == "Qwen/Qwen3-32B"
    assert settings.llm.base_url == "http://localhost:8001/v1"


def test_llm_features_config():
    """测试 LLM 功能配置"""
    settings = get_settings()
    assert settings.llm.features.semantic_segmentation is True
    assert settings.llm.features.error_correction is True
    assert settings.llm.features.expression_optimization is True


def test_llm_is_enabled():
    """测试 LLM 是否启用"""
    service = LLMService()
    assert service.is_enabled is True


@pytest.mark.asyncio
async def test_llm_optimize_empty_segments():
    """测试空段落优化"""
    service = LLMService()
    # 禁用 LLM 以测试直接返回
    service.settings.llm.enabled = False
    
    segments = []
    result = await service.optimize(segments)
    assert result == []


def test_segment_response_model():
    """测试段落响应模型"""
    segment = SegmentResponse(
        id=0,
        start=0.0,
        end=1.0,
        text="Hello world",
        speaker="SPEAKER_00",
    )
    assert segment.id == 0
    assert segment.text == "Hello world"
    assert segment.speaker == "SPEAKER_00"

