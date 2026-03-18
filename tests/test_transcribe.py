"""转录功能测试"""

import pytest
from fastapi.testclient import TestClient

from whisperlm.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    """健康检查接口返回必需字段"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "whisperx" in data
    assert "diarization" in data
    assert "llm" in data
    assert "gpu" in data


def test_health_llm_fields(client):
    """健康检查中 LLM 状态字段结构正确"""
    response = client.get("/health")
    data = response.json()
    llm = data["llm"]
    assert "provider" in llm
    assert "model" in llm
    assert "enabled" in llm


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "WhisperLM"


def test_transcribe_no_file(client):
    response = client.post("/api/v1/transcribe")
    assert response.status_code == 422  # Validation Error


def test_transcribe_unsupported_format(client):
    from io import BytesIO

    files = {"file": ("test.txt", BytesIO(b"test content"), "text/plain")}
    response = client.post("/api/v1/transcribe", files=files)
    assert response.status_code == 400
    assert "不支持的文件格式" in response.json()["detail"]


def test_legacy_transcribe_no_file(client):
    response = client.post("/transcribe/")
    assert response.status_code == 422


def test_transcribe_translate_no_llm(client):
    """LLM 未启用时 /transcribe-translate 返回 503"""
    from io import BytesIO

    files = {"file": ("test.mp3", BytesIO(b"fake audio"), "audio/mpeg")}
    data = {"target_language": "zh"}
    response = client.post("/api/v1/transcribe-translate", files=files, data=data)
    # LLM 默认关闭，应返回 503
    assert response.status_code == 503
