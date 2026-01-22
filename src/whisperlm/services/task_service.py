"""任务管理服务"""

import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from ..api.models import SegmentResponse, TranscribeResponse, WordTimestamp
from ..config import Settings, get_settings
from .whisperx_service import WhisperXService


class TaskService:
    """任务管理服务"""

    def __init__(self, settings: Settings | None = None,
                 whisperx_service: WhisperXService | None = None):
        self.settings = settings or get_settings()
        self.whisperx = whisperx_service or WhisperXService(self.settings)

    def _convert_segments(self, result: dict[str, Any]) -> list[SegmentResponse]:
        """转换 WhisperX 结果为标准格式"""
        segments = []
        for i, seg in enumerate(result.get("segments", [])):
            words = []
            for w in seg.get("words", []):
                words.append(WordTimestamp(
                    word=w.get("word", ""),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    confidence=w.get("score", 0.0),
                ))
            
            confidence = sum(w.confidence for w in words) / len(words) if words else 0.0
            segments.append(SegmentResponse(
                id=i,
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
                speaker=seg.get("speaker"),
                words=words,
                confidence=confidence,
            ))
        return segments

    def _get_duration(self, result: dict[str, Any], audio_path: Path) -> float:
        """获取音频时长（优先从结果中获取，避免重复加载音频）"""
        # 优先使用已计算的时长
        if "_audio_duration" in result:
            return result["_audio_duration"]
        
        # 回退：重新计算
        try:
            import whisperx
            audio = whisperx.load_audio(str(audio_path))
            return len(audio) / 16000
        except:
            return 0.0

    async def transcribe(self, audio_path: Path, language: str | None = None,
                        diarization: bool = True,
                        min_speakers: int | None = None, max_speakers: int | None = None) -> TranscribeResponse:
        """执行转录"""
        task_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        logger.info(f"[Task {task_id}] Starting transcription: {audio_path.name}")
        logger.info(f"[Task {task_id}] Parameters: language={language}, diarization={diarization}, "
                   f"speakers={min_speakers}-{max_speakers}")
        
        try:
            # WhisperX 转录
            whisperx_start = time.time()
            logger.info(f"[Task {task_id}] Starting WhisperX transcription...")
            result = self.whisperx.transcribe_complete(
                audio_path=audio_path,
                language=language,
                diarization=diarization,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            whisperx_time = time.time() - whisperx_start
            detected_language = result.get("language", "unknown")
            raw_segments_count = len(result.get("segments", []))
            logger.info(f"[Task {task_id}] WhisperX transcription completed: {whisperx_time:.2f}s, "
                       f"language={detected_language}, segments={raw_segments_count}")
            
            # 转换格式
            convert_start = time.time()
            segments = self._convert_segments(result)
            convert_time = time.time() - convert_start
            logger.info(f"[Task {task_id}] Format conversion completed: {convert_time:.2f}s, "
                       f"segments={len(segments)}")
            
            # 统计信息
            speakers = list(set(seg.speaker for seg in segments if seg.speaker))
            total_duration = self._get_duration(result, audio_path)
            total_time = time.time() - start_time
            
            logger.info(f"[Task {task_id}] Transcription completed: total_time={total_time:.2f}s, "
                       f"audio_duration={total_duration:.1f}s, "
                       f"segments={len(segments)}, speakers={len(speakers)}, "
                       f"speaker_list={speakers}")
            
            return TranscribeResponse(
                task_id=task_id,
                status="completed",
                language=detected_language,
                duration=total_duration,
                speakers=sorted(speakers),
                segments=segments,
            )
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"[Task {task_id}] Transcription failed: {total_time:.2f}s, error: {e}", exc_info=True)
            raise

