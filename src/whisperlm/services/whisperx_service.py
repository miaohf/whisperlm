"""WhisperX 核心转录服务"""

from pathlib import Path
from typing import Any

import torch
from loguru import logger

from ..config import Settings, get_settings


class WhisperXService:
    """WhisperX 转录服务"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model = None
        self._diarize_model = None
        # 缓存对齐模型 {language: (model, metadata) | None}
        # None 表示该语言的对齐模型加载失败
        self._align_models: dict[str, tuple[Any, Any] | None] = {}
        self._device = self.settings.whisperx.device
        self._compute_type = self.settings.whisperx.compute_type

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        """加载 Whisper 模型"""
        if self._model is not None:
            return

        import whisperx

        logger.info(f"Loading Whisper model: {self.settings.whisperx.model}")
        self._model = whisperx.load_model(
            self.settings.whisperx.model,
            self._device,
            compute_type=self._compute_type,
        )
        logger.info("Whisper model loaded")

    def load_diarization_model(self) -> None:
        """加载说话人分离模型"""
        if not self.settings.diarization.enabled:
            return
        if self._diarize_model is not None:
            return

        from whisperx.diarize import DiarizationPipeline

        hf_token = self.settings.diarization.huggingface_token
        if not hf_token:
            logger.warning("HF_TOKEN not configured, diarization will be unavailable")
            return

        logger.info("Loading diarization model...")
        self._diarize_model = DiarizationPipeline(
            use_auth_token=hf_token,
            device=self._device,
        )
        logger.info("Diarization model loaded")

    def load_audio(self, audio_path: str | Path) -> Any:
        """加载音频文件（单次加载，供多个步骤复用）"""
        import whisperx
        import time
        
        logger.debug(f"Loading audio: {Path(audio_path).name}")
        start_time = time.time()
        audio = whisperx.load_audio(str(audio_path))
        load_time = time.time() - start_time
        audio_duration = len(audio) / 16000
        logger.info(f"Audio loaded: {load_time:.2f}s, duration={audio_duration:.2f}s, samples={len(audio)}")
        return audio

    def transcribe(self, audio: Any, audio_name: str = "audio", language: str | None = None) -> dict[str, Any]:
        """执行转录（接收已加载的音频数据）"""
        import time

        if self._model is None:
            self.load_model()

        audio_duration = len(audio) / 16000
        logger.info(f"Starting Whisper transcription: {audio_name}, language={language or 'auto'}")
        start_time = time.time()
        
        result = self._model.transcribe(
            audio,
            batch_size=self.settings.whisperx.batch_size,
            language=language or self.settings.whisperx.language,
        )
        
        transcribe_time = time.time() - start_time
        detected_lang = result.get("language", "unknown")
        segments_count = len(result.get("segments", []))
        logger.info(f"Whisper transcription completed: {transcribe_time:.2f}s, "
                   f"speed {audio_duration/transcribe_time:.2f}x, "
                   f"language={detected_lang}, segments={segments_count}")
        
        return result

    def _get_align_model(self, language: str) -> tuple[Any, Any] | None:
        """获取对齐模型（带缓存），失败时返回 None"""
        import whisperx
        
        # 如果之前加载失败，直接返回 None
        if language in self._align_models and self._align_models[language] is None:
            return None
        
        if language not in self._align_models:
            logger.info(f"Loading alignment model for language: {language}")
            
            # 检查配置中是否指定了该语言的对齐模型
            model_name = None
            if self.settings.whisperx.align_models and language in self.settings.whisperx.align_models:
                model_name = self.settings.whisperx.align_models[language]
                logger.info(f"Using configured alignment model for {language}: {model_name}")
            
            try:
                # 如果指定了模型名，使用它；否则使用默认模型
                if model_name:
                    model_a, metadata = whisperx.load_align_model(
                        language_code=language,
                        device=self._device,
                        model_name=model_name
                    )
                else:
                    model_a, metadata = whisperx.load_align_model(
                        language_code=language,
                        device=self._device
                    )
                self._align_models[language] = (model_a, metadata)
                logger.info(f"Alignment model loaded for: {language}")
            except Exception as e:
                logger.warning(f"Failed to load alignment model for {language}: {e}")
                logger.warning(f"Word-level alignment will be skipped for {language}. Transcription will continue without word timestamps.")
                # 如果配置了模型但加载失败，尝试使用默认模型
                if model_name:
                    logger.info(f"Trying default alignment model for {language}...")
                    try:
                        model_a, metadata = whisperx.load_align_model(
                            language_code=language,
                            device=self._device
                        )
                        self._align_models[language] = (model_a, metadata)
                        logger.info(f"Default alignment model loaded for: {language}")
                    except Exception as e2:
                        logger.warning(f"Default alignment model also failed: {e2}")
                        self._align_models[language] = None
                        return None
                else:
                    # 缓存失败标记，避免重复尝试
                    self._align_models[language] = None
                    return None
        
        return self._align_models[language]

    def align(self, result: dict[str, Any], audio: Any) -> dict[str, Any]:
        """执行词级对齐（接收已加载的音频数据），如果模型不可用则跳过"""
        import whisperx
        import time

        language = result.get("language", "en")
        logger.info(f"Starting word-level alignment: language={language}")
        start_time = time.time()
        
        try:
            align_model = self._get_align_model(language)
            if align_model is None:
                logger.warning(f"Alignment model not available for {language}, skipping word-level alignment")
                logger.info("Transcription will continue without word timestamps")
                # 确保结果中有 segments，即使没有 words
                if "segments" not in result:
                    result["segments"] = []
                return result
            
            model_a, metadata = align_model
            result = whisperx.align(result["segments"], model_a, metadata, audio, self._device)
            
            align_time = time.time() - start_time
            words_count = sum(len(seg.get("words", [])) for seg in result.get("segments", []))
            logger.info(f"Word-level alignment completed: {align_time:.2f}s, words={words_count}")
        except Exception as e:
            # 对齐失败不应该影响转录，记录警告并继续
            align_time = time.time() - start_time
            logger.warning(f"Word-level alignment failed after {align_time:.2f}s: {e}")
            logger.warning(f"Transcription will continue without word timestamps for language: {language}")
            # 确保结果中有 segments，即使没有 words
            if "segments" not in result:
                result["segments"] = []
        
        return result

    def diarize(self, result: dict[str, Any], audio: Any,
                min_speakers: int | None = None, max_speakers: int | None = None) -> dict[str, Any]:
        """执行说话人分离（接收已加载的音频数据）"""
        import whisperx
        import time

        if self._diarize_model is None:
            logger.warning("Diarization model not loaded, skipping")
            return result

        logger.info(f"Starting speaker diarization: min_speakers={min_speakers}, max_speakers={max_speakers}")
        start_time = time.time()
        
        diarize_segments = self._diarize_model(
            audio,
            min_speakers=min_speakers or self.settings.diarization.min_speakers,
            max_speakers=max_speakers or self.settings.diarization.max_speakers,
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        diarize_time = time.time() - start_time
        speakers = set(seg.get("speaker") for seg in result.get("segments", []) if seg.get("speaker"))
        logger.info(f"Speaker diarization completed: {diarize_time:.2f}s, detected_speakers={sorted(speakers)}")
        
        return result

    def transcribe_complete(self, audio_path: str | Path, language: str | None = None,
                           diarization: bool = True, min_speakers: int | None = None,
                           max_speakers: int | None = None) -> dict[str, Any]:
        """完整的转录流程（音频只加载一次）"""
        import time
        
        total_start = time.time()
        audio_name = Path(audio_path).name
        logger.info(f"Starting complete transcription pipeline: {audio_name}")
        
        # 0. 加载音频（只加载一次，后续步骤复用）
        audio = self.load_audio(audio_path)
        audio_duration = len(audio) / 16000
        
        # 1. 转录
        result = self.transcribe(audio, audio_name, language)
        detected_language = result.get("language", language or "en")
        
        # 2. 词级对齐（失败不影响转录）
        try:
            result = self.align(result, audio)
        except Exception as e:
            logger.warning(f"Alignment step failed but transcription will continue: {e}")
            # 确保结果结构完整
            if "segments" not in result:
                result["segments"] = []
        
        result["language"] = detected_language
        
        # 3. 说话人分离
        if diarization and self.settings.diarization.enabled:
            result = self.diarize(result, audio, min_speakers, max_speakers)
        else:
            logger.info("Diarization disabled, skipping")
        
        total_time = time.time() - total_start
        segments_count = len(result.get("segments", []))
        speed = audio_duration / total_time if total_time > 0 else 0
        logger.info(f"Complete transcription pipeline completed: {total_time:.2f}s, "
                   f"audio={audio_duration:.1f}s, speed={speed:.2f}x, segments={segments_count}")
        
        # 将音频时长附加到结果中，避免后续再次加载
        result["_audio_duration"] = audio_duration
        
        return result

    def get_gpu_info(self) -> dict[str, Any]:
        """获取 GPU 信息"""
        if not torch.cuda.is_available():
            return {"available": False, "name": None, "memory_total": None, "memory_used": None}

        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
            "memory_used": f"{torch.cuda.memory_allocated(0) / 1024**3:.1f}GB",
        }

