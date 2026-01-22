"""API 路由"""

import tempfile
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from fastapi.responses import StreamingResponse
from loguru import logger

from ..config import Settings, get_settings
from ..services.task_service import TaskService
from ..services.whisperx_service import WhisperXService
from ..services.separation_service import SeparationService
from .models import TranscribeResponse, HealthResponse, WhisperXStatus, DiarizationStatus, GPUInfo

router = APIRouter(prefix="/api/v1", tags=["transcribe"])

_separation_service: SeparationService | None = None


def get_separation_service() -> SeparationService:
    """获取音频分离服务实例"""
    global _separation_service
    if _separation_service is None:
        settings = get_settings()
        _separation_service = SeparationService(
            device=settings.whisperx.device,
            model="htdemucs",
        )
    return _separation_service

_task_service: TaskService | None = None


def get_task_service() -> TaskService:
    global _task_service
    if _task_service is None:
        _task_service = TaskService()
    return _task_service


SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4", ".mkv", ".avi", ".mov", ".webm"}


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: Annotated[UploadFile, File(description="音频/视频文件")],
    language: Annotated[str | None, Form(description="语言代码")] = None,
    diarization: Annotated[bool, Form(description="是否启用说话人分离")] = True,
    min_speakers: Annotated[int | None, Form(description="最小说话人数")] = None,
    max_speakers: Annotated[int | None, Form(description="最大说话人数")] = None,
    task_service: TaskService = Depends(get_task_service),
):
    """转录音频/视频文件"""
    import time
    
    request_start = time.time()
    
    # 立即打印请求简要信息（在读取文件之前）
    logger.info(f"[API] Received transcription request: file={file.filename}, "
               f"language={language}, diarization={diarization}, "
               f"speakers={min_speakers}-{max_speakers}")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {suffix}")

    if language == "auto":
        language = None

    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        file_size = len(content)
        tmp.write(content)
        audio_path = Path(tmp.name)
        logger.debug(f"[API] File saved to temporary path: {audio_path}, size: {file_size} bytes")

    try:
        response = await task_service.transcribe(
            audio_path=audio_path,
            language=language,
            diarization=diarization,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        
        request_time = time.time() - request_start
        logger.info(f"[API] Transcription request completed: task_id={response.task_id}, total_time={request_time:.2f}s, "
                   f"segments={len(response.segments)}, speakers={len(response.speakers)}")
        
        return response
    except Exception as e:
        request_time = time.time() - request_start
        logger.error(f"[API] Transcription request failed: {request_time:.2f}s, error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"转录失败: {str(e)}")
    finally:
        audio_path.unlink(missing_ok=True)
        logger.debug(f"[API] Temporary file cleaned: {audio_path}")




async def generate_multipart_response(vocals_path: Path, background_path: Path):
    """生成 multipart/form-data 响应流"""
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"

    # 写入人声文件
    yield f"--{boundary}\r\n".encode()
    yield f'Content-Disposition: form-data; name="vocals"; filename="{vocals_path.name}"\r\n'.encode()
    yield b"Content-Type: audio/wav\r\n\r\n"
    async with aiofiles.open(vocals_path, "rb") as f:
        while chunk := await f.read(8192):
            yield chunk
    yield b"\r\n"

    # 写入背景音文件
    yield f"--{boundary}\r\n".encode()
    yield f'Content-Disposition: form-data; name="background"; filename="{background_path.name}"\r\n'.encode()
    yield b"Content-Type: audio/wav\r\n\r\n"
    async with aiofiles.open(background_path, "rb") as f:
        while chunk := await f.read(8192):
            yield chunk
    yield b"\r\n"

    # 结束边界
    yield f"--{boundary}--\r\n".encode()


@router.post("/separate")
async def separate_audio(
    file: Annotated[UploadFile, File(description="音频文件")],
    model: Annotated[str, Form(description="分离模型")] = "htdemucs",
    separation_service: SeparationService = Depends(get_separation_service),
):
    """
    分离音频为人声和背景音
    
    返回格式：multipart/form-data
    - vocals: 人声文件流
    - background: 背景音文件流
    """
    import time
    
    request_start = time.time()
    
    # 立即打印请求简要信息（在读取文件之前）
    logger.info(f"[API] Received audio separation request: file={file.filename}, model={model}")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {suffix}")

    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        file_size = len(content)
        tmp.write(content)
        audio_path = Path(tmp.name)
        logger.debug(f"[API] File saved: {audio_path}, size: {file_size} bytes")

    output_dir = Path(tempfile.mkdtemp())
    try:
        # 分离音频
        separate_start = time.time()
        vocals_path, background_path = separation_service.separate(audio_path, output_dir)
        separate_time = time.time() - separate_start
        
        vocals_size = vocals_path.stat().st_size if vocals_path.exists() else 0
        background_size = background_path.stat().st_size if background_path.exists() else 0
        
        logger.info(f"[API] Audio separation completed: {separate_time:.2f}s, "
                   f"vocals={vocals_size} bytes, background={background_size} bytes")

        # 生成 multipart 响应流
        response_stream = generate_multipart_response(vocals_path, background_path)

        request_time = time.time() - request_start
        logger.info(f"[API] Audio separation request completed: total_time={request_time:.2f}s")

        return StreamingResponse(
            response_stream,
            media_type="multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        )
    except Exception as e:
        request_time = time.time() - request_start
        logger.error(f"[API] Audio separation request failed: {request_time:.2f}s, error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"音频分离失败: {str(e)}")
    finally:
        # 清理临时文件
        audio_path.unlink(missing_ok=True)
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
            logger.debug(f"[API] Temporary directory cleaned: {output_dir}")


def init_services(settings: Settings | None = None) -> None:
    """初始化服务"""
    global _task_service

    settings = settings or get_settings()
    whisperx_service = WhisperXService(settings)

    logger.info("Preloading models...")
    whisperx_service.load_model()
    whisperx_service.load_diarization_model()

    _task_service = TaskService(
        settings=settings,
        whisperx_service=whisperx_service,
    )
    logger.info("Service initialization completed")


def get_health_status(settings: Settings | None = None) -> HealthResponse:
    """获取健康状态"""
    from .. import __version__

    settings = settings or get_settings()
    whisperx_service = _task_service.whisperx if _task_service else WhisperXService(settings)

    gpu_info = whisperx_service.get_gpu_info()

    return HealthResponse(
        status="healthy",
        version=__version__,
        whisperx=WhisperXStatus(
            model=settings.whisperx.model,
            device=settings.whisperx.device,
            loaded=whisperx_service.is_loaded,
        ),
        diarization=DiarizationStatus(loaded=settings.diarization.enabled),
        gpu=GPUInfo(**gpu_info),
    )

