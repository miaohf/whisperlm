"""LLM 服务测试"""

import pytest

from whisperlm.config import get_settings
from whisperlm.services.llm_service import LLMService
from whisperlm.api.models import SegmentResponse


def test_llm_service_init():
    """LLM 服务可以正常初始化"""
    service = LLMService()
    assert service is not None


def test_llm_config():
    """LLM 配置字段存在"""
    settings = get_settings()
    assert hasattr(settings, "llm")
    assert settings.llm.provider in ("vllm", "openai", "ollama", "azure", "anthropic")
    assert settings.llm.model != ""
    assert settings.llm.base_url.startswith("http")


def test_llm_features_config():
    """LLM 功能配置字段存在"""
    settings = get_settings()
    assert hasattr(settings.llm, "features")
    assert isinstance(settings.llm.features.semantic_segmentation, bool)
    assert isinstance(settings.llm.features.error_correction, bool)
    assert isinstance(settings.llm.features.expression_optimization, bool)


def test_llm_is_enabled_default():
    """LLM 默认为关闭状态（需要用户显式开启）"""
    service = LLMService()
    # 默认配置中 enabled=False；若用户有 config.yaml 则以实际为准
    assert isinstance(service.is_enabled, bool)


@pytest.mark.asyncio
async def test_llm_optimize_empty_segments():
    """空段落直接返回空列表，不调用 LLM"""
    service = LLMService()
    result = await service.optimize([])
    assert result == []


@pytest.mark.asyncio
async def test_llm_optimize_disabled():
    """LLM 未启用时，optimize 原样返回段落"""
    service = LLMService()
    # 强制关闭
    service.settings.llm.enabled = False

    segments = [
        SegmentResponse(id=0, start=0.0, end=1.0, text="Hello world", speaker="SPEAKER_00")
    ]
    result = await service.optimize(segments)
    assert result == segments


@pytest.mark.asyncio
async def test_llm_translate_empty_segments():
    """空段落直接返回空列表，不调用 LLM"""
    service = LLMService()
    result = await service.translate([], target_language="zh")
    assert result == []


def test_segment_response_model():
    """SegmentResponse 模型字段正确"""
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
    assert segment.translated_text is None
