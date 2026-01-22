"""FastAPI 应用入口"""

import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api.routes import router, init_services, get_health_status
from .api.legacy_routes import legacy_router
from .api.models import HealthResponse
from .config import get_settings, load_config
from .utils.logger import logger, setup_logger

# 设置 Python 无缓冲模式（确保日志实时输出）
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# 设置 stdout 为行缓冲模式
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# 初始化日志配置
setup_logger(level="INFO", enable_file=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info(f"WhisperLM v{__version__} starting...")

    settings = get_settings()
    logger.info(f"Config: Whisper model={settings.whisperx.model}, "
                f"device={settings.whisperx.device}, "
                f"compute_type={settings.whisperx.compute_type}")

    try:
        init_services(settings)
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        raise

    logger.info("WhisperLM startup completed")
    yield
    logger.info("WhisperLM shutting down...")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    settings = load_config()

    app = FastAPI(
        title="WhisperLM",
        description="智能语音转文字服务：WhisperX 精确转录 + 说话人分离",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    app.include_router(legacy_router)

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check():
        return get_health_status(settings)

    @app.get("/", tags=["health"])
    async def root():
        return {"service": "WhisperLM", "version": __version__, "docs": "/docs"}

    return app


app = create_app()


def main():
    """主入口函数"""
    settings = get_settings()
    logger.info(f"Starting service: http://{settings.server.host}:{settings.server.port}")
    uvicorn.run(
        "whisperlm.main:app",
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers,
        reload=False,
        log_config=None,  # 禁用 uvicorn 的默认日志配置，使用我们的 loguru
        access_log=False,  # 禁用访问日志，避免干扰
    )


if __name__ == "__main__":
    main()

