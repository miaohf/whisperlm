"""Pydantic 数据模型"""

from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    """词级时间戳"""
    word: str
    start: float
    end: float
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SegmentResponse(BaseModel):
    """转录段落响应"""
    id: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[WordTimestamp] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    translated_text: str | None = None


class TranscribeResponse(BaseModel):
    """转录响应"""
    task_id: str
    status: str = "completed"
    language: str | None = None
    duration: float = 0.0
    speakers: list[str] = Field(default_factory=list)
    segments: list[SegmentResponse] = Field(default_factory=list)


class LegacySegment(BaseModel):
    """旧版段落格式"""
    start: float
    end: float
    text: str
    speaker: str | None = None


class LegacyTranscribeResponse(BaseModel):
    """旧版转录响应"""
    status: str = "success"
    segments: list[LegacySegment] = Field(default_factory=list)


class GPUInfo(BaseModel):
    """GPU 信息"""
    available: bool
    name: str | None = None
    memory_total: str | None = None
    memory_used: str | None = None


class WhisperXStatus(BaseModel):
    """WhisperX 状态"""
    model: str
    device: str
    loaded: bool


class DiarizationStatus(BaseModel):
    """说话人分离状态"""
    model: str = "pyannote/speaker-diarization-3.1"
    loaded: bool


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "healthy"
    version: str
    whisperx: WhisperXStatus
    diarization: DiarizationStatus
    gpu: GPUInfo

