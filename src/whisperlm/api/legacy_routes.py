"""兼容旧版接口"""

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from loguru import logger

from ..services.task_service import TaskService
from .models import LegacySegment, LegacyTranscribeResponse
from .routes import get_task_service, SUPPORTED_FORMATS

legacy_router = APIRouter(tags=["legacy"])


@legacy_router.post("/transcribe/", response_model=LegacyTranscribeResponse)
async def legacy_transcribe(
    file: Annotated[UploadFile, File(description="音频/视频文件")],
    task_service: TaskService = Depends(get_task_service),
):
    """兼容旧版转录接口"""
    logger.info(f"[Legacy API] Received transcription request: file={file.filename}")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        audio_path = Path(tmp.name)

    try:
        response = await task_service.transcribe(
            audio_path=audio_path,
            language=None,
            diarization=True,
        )

        segments = [
            LegacySegment(start=seg.start, end=seg.end, text=seg.text, speaker=seg.speaker)
            for seg in response.segments
        ]

        return LegacyTranscribeResponse(status="success", segments=segments)

    except Exception as e:
        logger.error(f"转录失败: {e}")
        return LegacyTranscribeResponse(status="error", segments=[])

    finally:
        audio_path.unlink(missing_ok=True)

