"""音频分离服务（Demucs）"""

import tempfile
import time
from pathlib import Path

import torch
from loguru import logger


class SeparationService:
    """音频分离服务（使用 Demucs）"""

    def __init__(self, device: str = "cuda", model: str = "htdemucs"):
        self.device = device
        self.model_name = model
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        """加载 Demucs 模型"""
        if self._model is not None:
            return

        try:
            from demucs.pretrained import get_model
            logger.info(f"Loading Demucs model: {self.model_name}")
            self._model = get_model(self.model_name)
            self._model.to(self.device)
            logger.info("Demucs model loaded")
        except ImportError:
            logger.error("demucs not installed, please run: pip install demucs")
            raise
        except Exception as e:
            logger.error(f"Failed to load Demucs model: {e}")
            raise

    def separate(self, audio_path: str | Path, output_dir: Path | None = None) -> tuple[Path, Path]:
        """
        分离音频为人声和背景音
        
        Args:
            audio_path: 输入音频文件路径
            output_dir: 输出目录，如果为 None 则使用临时目录
            
        Returns:
            (vocals_path, background_path): 人声和背景音文件路径
        """
        if self._model is None:
            self.load_model()

        from demucs.apply import apply_model
        from demucs.audio import AudioFile, convert_audio
        import soundfile as sf

        audio_path = Path(audio_path)
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting audio separation: {audio_path.name}")
        start_time = time.time()

        # 加载音频
        load_start = time.time()
        wav = AudioFile(str(audio_path)).read(streams=0, samplerate=self._model.sample_rate, channels=2)
        wav = convert_audio(wav, self._model.sample_rate, self._model.channels, self._model.sample_rate)
        audio_duration = wav.shape[-1] / self._model.sample_rate
        load_time = time.time() - load_start
        logger.debug(f"Audio loading completed: {load_time:.2f}s, duration: {audio_duration:.2f}s")
        
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        wav = wav[None]  # 添加 batch 维度

        # 分离
        separate_start = time.time()
        logger.info("Starting Demucs separation processing...")
        with torch.no_grad():
            sources = apply_model(self._model, wav.to(self.device), shifts=1, split=True, overlap=0.25, progress=False)
            sources = sources[0]  # 移除 batch 维度
            sources = sources * ref.std() + ref.mean()
        separate_time = time.time() - separate_start
        logger.info(f"Demucs separation completed: {separate_time:.2f}s, speed {audio_duration/separate_time:.2f}x")

        # Demucs 输出顺序: [drums, bass, other, vocals]
        vocals = sources[3].cpu()  # vocals
        background = (sources[0] + sources[1] + sources[2]).cpu()  # drums + bass + other

        # 保存文件
        save_start = time.time()
        vocals_path = output_dir / f"{audio_path.stem}_vocals.wav"
        background_path = output_dir / f"{audio_path.stem}_background.wav"

        # 转换为 numpy 并保存 (sources shape: [channels, samples])
        vocals_np = vocals.numpy().T  # [samples, channels]
        background_np = background.numpy().T

        sf.write(str(vocals_path), vocals_np, self._model.sample_rate)
        sf.write(str(background_path), background_np, self._model.sample_rate)
        save_time = time.time() - save_start
        
        total_time = time.time() - start_time
        vocals_size = vocals_path.stat().st_size
        background_size = background_path.stat().st_size
        
        logger.info(f"Audio separation completed: total_time={total_time:.2f}s (loading {load_time:.2f}s, "
                   f"separation {separate_time:.2f}s, saving {save_time:.2f}s), "
                   f"vocals={vocals_size/1024/1024:.2f}MB, background={background_size/1024/1024:.2f}MB")
        
        return vocals_path, background_path

