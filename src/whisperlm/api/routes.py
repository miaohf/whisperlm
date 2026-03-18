"""API 路由"""

import tempfile
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from ..config import Settings, get_settings
from ..services.task_service import TaskService
from ..services.whisperx_service import WhisperXService
from ..services.separation_service import SeparationService
from ..services.llm_service import LLMService
from .models import (
    TranscribeResponse,
    HealthResponse,
    WhisperXStatus,
    DiarizationStatus,
    LLMStatus,
    GPUInfo,
)

router = APIRouter(prefix="/api/v1", tags=["transcribe"])

SUPPORTED_FORMATS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
}


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 / 1024:.1f}MB"


# ─────────────────────────────────────────────────────────────────────────────
# 服务单例（在 init_services 中初始化）
# ─────────────────────────────────────────────────────────────────────────────

_task_service: TaskService | None = None
_llm_service: LLMService | None = None
_separation_service: SeparationService | None = None


def get_task_service() -> TaskService:
    global _task_service
    if _task_service is None:
        _task_service = TaskService()
    return _task_service


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(get_settings())
    return _llm_service


def get_separation_service() -> SeparationService:
    global _separation_service
    if _separation_service is None:
        settings = get_settings()
        _separation_service = SeparationService(
            device=settings.whisperx.device,
            model="htdemucs",
        )
    return _separation_service


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

async def _save_upload(file: UploadFile) -> tuple[Path, int]:
    """将上传文件保存到临时文件，返回 (path, size)"""
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        return Path(tmp.name), len(content)


def _validate_file(file: UploadFile) -> str:
    """校验文件名与格式，返回后缀；不合法则抛 HTTPException"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {suffix}")
    return suffix


# ─────────────────────────────────────────────────────────────────────────────
# 路由：转录
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    request: Request,
    file: Annotated[UploadFile, File(description="音频/视频文件")],
    language: Annotated[str | None, Form(description="语言代码，留空自动检测")] = None,
    diarization: Annotated[bool, Form(description="是否启用说话人分离")] = True,
    min_speakers: Annotated[int | None, Form(description="最小说话人数")] = None,
    max_speakers: Annotated[int | None, Form(description="最大说话人数")] = None,
    llm_optimize: Annotated[bool, Form(description="是否启用 LLM 语义优化")] = True,
    task_service: TaskService = Depends(get_task_service),
    llm_service: LLMService = Depends(get_llm_service),
):
    """转录音频/视频文件，可选 LLM 语义优化"""
    import time
    request_start = time.time()

    client_ip = request.client.host if request.client else "unknown"
    logger.info("[API] ======== Request Start ========")
    logger.info(f"[API] Client: {client_ip}")
    logger.info(f"[API] Endpoint: {request.method} {request.url.path}")
    logger.info(f"[API] File: name={file.filename}, content_type={file.content_type}")
    logger.info(
        f"[API] Parameters: language={language}, diarization={diarization}, "
        f"speakers={min_speakers}-{max_speakers}, llm_optimize={llm_optimize}"
    )

    _validate_file(file)
    if language == "auto":
        language = None

    audio_path, file_size = await _save_upload(file)
    logger.info(f"[API] File size: {_format_size(file_size)}")
    logger.info("[API] ======== Request End ========")

    try:
        response = await task_service.transcribe(
            audio_path=audio_path,
            language=language,
            diarization=diarization,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        # LLM 语义优化（可选）
        if llm_optimize and llm_service.is_enabled:
            response.segments = await llm_service.optimize(
                segments=response.segments,
                language=response.language,
            )

        elapsed = time.time() - request_start
        logger.info(
            f"[API] Transcription completed: task_id={response.task_id}, "
            f"total_time={elapsed:.2f}s, segments={len(response.segments)}, "
            f"speakers={len(response.speakers)}"
        )
        return response

    except Exception as e:
        elapsed = time.time() - request_start
        logger.error(f"[API] Transcription failed after {elapsed:.2f}s: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"转录失败: {str(e)}")
    finally:
        audio_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 路由：转录 + 翻译
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/transcribe-translate", response_model=TranscribeResponse)
async def transcribe_translate(
    request: Request,
    file: Annotated[UploadFile, File(description="音频/视频文件")],
    target_language: Annotated[str, Form(description="目标语言代码，如 zh、en、ja")],
    source_language: Annotated[str | None, Form(description="原始语言代码，留空自动检测")] = None,
    diarization: Annotated[bool, Form(description="是否启用说话人分离")] = True,
    min_speakers: Annotated[int | None, Form(description="最小说话人数")] = None,
    max_speakers: Annotated[int | None, Form(description="最大说话人数")] = None,
    llm_optimize: Annotated[bool, Form(description="翻译前是否先进行 LLM 语义优化")] = False,
    translation_style: Annotated[str, Form(description="翻译风格：natural / formal / casual")] = "natural",
    task_service: TaskService = Depends(get_task_service),
    llm_service: LLMService = Depends(get_llm_service),
):
    """转录后直接使用 LLM 翻译，segments 中包含 translated_text 字段"""
    import time
    request_start = time.time()

    client_ip = request.client.host if request.client else "unknown"
    logger.info("[API] ======== Request Start ========")
    logger.info(f"[API] Client: {client_ip}")
    logger.info(f"[API] Endpoint: {request.method} {request.url.path}")
    logger.info(
        f"[API] File: name={file.filename}  "
        f"source={source_language or 'auto'} -> target={target_language}, "
        f"style={translation_style}"
    )

    if not llm_service.is_enabled:
        raise HTTPException(
            status_code=503,
            detail="LLM 服务未启用，请在 config.yaml 中设置 llm.enabled: true",
        )

    _validate_file(file)
    if source_language == "auto":
        source_language = None

    audio_path, file_size = await _save_upload(file)
    logger.info(f"[API] File size: {_format_size(file_size)}")
    logger.info("[API] ======== Request End ========")

    try:
        # 1. 转录
        response = await task_service.transcribe(
            audio_path=audio_path,
            language=source_language,
            diarization=diarization,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        # 2. （可选）LLM 语义优化
        if llm_optimize:
            response.segments = await llm_service.optimize(
                segments=response.segments,
                language=response.language,
            )

        # 3. LLM 翻译
        response.segments = await llm_service.translate(
            segments=response.segments,
            target_language=target_language,
            style=translation_style,
        )

        # 4. 补充翻译相关元数据
        response.source_language = response.language
        response.target_language = target_language

        elapsed = time.time() - request_start
        logger.info(
            f"[API] Transcribe-translate completed: task_id={response.task_id}, "
            f"total_time={elapsed:.2f}s, segments={len(response.segments)}"
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        elapsed = time.time() - request_start
        logger.error(
            f"[API] Transcribe-translate failed after {elapsed:.2f}s: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"转录翻译失败: {str(e)}")
    finally:
        audio_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 路由：音频人声/背景分离
# ─────────────────────────────────────────────────────────────────────────────

async def _generate_multipart_response(vocals_path: Path, background_path: Path):
    """生成 multipart/form-data 响应流"""
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"

    yield f"--{boundary}\r\n".encode()
    yield f'Content-Disposition: form-data; name="vocals"; filename="{vocals_path.name}"\r\n'.encode()
    yield b"Content-Type: audio/wav\r\n\r\n"
    async with aiofiles.open(vocals_path, "rb") as f:
        while chunk := await f.read(8192):
            yield chunk
    yield b"\r\n"

    yield f"--{boundary}\r\n".encode()
    yield f'Content-Disposition: form-data; name="background"; filename="{background_path.name}"\r\n'.encode()
    yield b"Content-Type: audio/wav\r\n\r\n"
    async with aiofiles.open(background_path, "rb") as f:
        while chunk := await f.read(8192):
            yield chunk
    yield b"\r\n"

    yield f"--{boundary}--\r\n".encode()


@router.post("/separate")
async def separate_audio(
    request: Request,
    file: Annotated[UploadFile, File(description="音频文件")],
    model: Annotated[str, Form(description="分离模型")] = "htdemucs",
    separation_service: SeparationService = Depends(get_separation_service),
):
    """
    分离音频为人声和背景音。

    返回格式：multipart/form-data
    - vocals：人声
    - background：背景音（鼓 + 贝斯 + 其他）
    """
    import time
    request_start = time.time()

    client_ip = request.client.host if request.client else "unknown"
    logger.info("[API] ======== Request Start ========")
    logger.info(f"[API] Client: {client_ip}")
    logger.info(f"[API] Endpoint: {request.method} {request.url.path}")
    logger.info(f"[API] File: name={file.filename}, model={model}")

    _validate_file(file)
    audio_path, file_size = await _save_upload(file)
    logger.info(f"[API] File size: {_format_size(file_size)}")
    logger.info("[API] ======== Request End ========")

    output_dir = Path(tempfile.mkdtemp())
    try:
        sep_start = time.time()
        vocals_path, background_path = separation_service.separate(audio_path, output_dir)
        sep_time = time.time() - sep_start

        vocals_size = vocals_path.stat().st_size if vocals_path.exists() else 0
        background_size = background_path.stat().st_size if background_path.exists() else 0
        logger.info(
            f"[API] Audio separation completed: {sep_time:.2f}s, "
            f"vocals={vocals_size} bytes, background={background_size} bytes"
        )

        elapsed = time.time() - request_start
        logger.info(f"[API] Separate request completed: total_time={elapsed:.2f}s")

        return StreamingResponse(
            _generate_multipart_response(vocals_path, background_path),
            media_type="multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        )
    except Exception as e:
        elapsed = time.time() - request_start
        logger.error(f"[API] Audio separation failed after {elapsed:.2f}s: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"音频分离失败: {str(e)}")
    finally:
        audio_path.unlink(missing_ok=True)
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 服务初始化与健康状态（供 main.py 调用）
# ─────────────────────────────────────────────────────────────────────────────

def init_services(settings: Settings | None = None) -> None:
    """应用启动时预热所有服务"""
    global _task_service, _llm_service

    settings = settings or get_settings()

    whisperx_service = WhisperXService(settings)
    logger.info("Preloading models...")
    whisperx_service.load_model()
    whisperx_service.load_diarization_model()

    _task_service = TaskService(
        settings=settings,
        whisperx_service=whisperx_service,
    )

    _llm_service = LLMService(settings)
    llm_status = "enabled" if settings.llm.enabled else "disabled"
    logger.info(f"LLM service initialized: {llm_status} (provider={settings.llm.provider})")

    logger.info("Service initialization completed")


def get_health_status(settings: Settings | None = None) -> HealthResponse:
    """构建健康状态响应（同步部分，LLM 连接状态由调用方异步补充）"""
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
        diarization=DiarizationStatus(
            loaded=settings.diarization.enabled,
        ),
        llm=LLMStatus(
            provider=settings.llm.provider,
            model=settings.llm.model,
            enabled=settings.llm.enabled,
            connected=None,  # 由调用方异步检测后填充
        ),
        gpu=GPUInfo(**gpu_info),
    )
