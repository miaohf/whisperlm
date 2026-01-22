"""转录功能测试"""

import pytest
from fastapi.testclient import TestClient

from whisperlm.main import app


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


def test_health_check(client):
    """测试健康检查接口"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "whisperx" in data
    assert "llm" in data


def test_root(client):
    """测试根路径"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "WhisperLM"


def test_transcribe_no_file(client):
    """测试无文件上传"""
    response = client.post("/api/v1/transcribe")
    assert response.status_code == 422  # Validation Error


def test_transcribe_unsupported_format(client):
    """测试不支持的文件格式"""
    from io import BytesIO

    files = {"file": ("test.txt", BytesIO(b"test content"), "text/plain")}
    response = client.post("/api/v1/transcribe", files=files)
    assert response.status_code == 400
    assert "不支持的文件格式" in response.json()["detail"]


def test_legacy_transcribe_no_file(client):
    """测试旧版接口无文件上传"""
    response = client.post("/transcribe/")
    assert response.status_code == 422

