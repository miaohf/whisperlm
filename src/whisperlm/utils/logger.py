"""日志配置模块"""

import sys
from pathlib import Path

from loguru import logger

# 缓存项目根目录
_PROJECT_ROOT: Path | None = None


def _find_project_root() -> Path | None:
    """查找项目根目录"""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT
    
    # 从当前文件向上查找
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
            _PROJECT_ROOT = parent
            return _PROJECT_ROOT
    return None


def _get_relative_path(file_path: str) -> str:
    """获取相对于项目根目录的路径"""
    try:
        project_root = _find_project_root()
        if project_root:
            current = Path(file_path).resolve()
            try:
                path = str(current.relative_to(project_root))
            except ValueError:
                path = Path(file_path).name
        else:
            path = Path(file_path).name
    except Exception:
        path = Path(file_path).name
    
    # 转义路径中的尖括号，避免被 loguru 误认为颜色标签
    return path.replace("<", r"\<").replace(">", r"\>")


def _format_record(record: dict) -> str:
    """自定义格式化函数，支持相对路径"""
    # 获取相对路径
    relative_path = _get_relative_path(record["file"].path)
    
    # 根据日志级别设置颜色
    level_colors = {
        "TRACE": "<dim>",
        "DEBUG": "<blue>",
        "INFO": "<green>",
        "SUCCESS": "<bold><green>",
        "WARNING": "<yellow>",
        "ERROR": "<red>",
        "CRITICAL": "<bold><red>",
    }
    level_color = level_colors.get(record["level"].name, "")
    level_end = "</>" if level_color else ""
    
    # 构建格式化字符串
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        f"{level_color}{{level: <8}}{level_end} | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
        f"<blue>{relative_path}</blue>:<cyan>{{line}}</cyan> | "
        f"{level_color}{{message}}{level_end}"
    )
    
    # 添加异常信息
    if record["exception"]:
        fmt += "\n{exception}"
    
    return fmt + "\n"


def _format_record_file(record: dict) -> str:
    """文件日志格式化函数（无颜色）"""
    relative_path = _get_relative_path(record["file"].path)
    
    fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function} | "
        f"{relative_path}:{{line}} | "
        "{message}"
    )
    
    if record["exception"]:
        fmt += "\n{exception}"
    
    return fmt + "\n"


class UnbufferedStderr:
    """无缓冲的 stderr 包装器，直接写入文件描述符"""
    def __init__(self):
        import os
        self._fd = sys.stderr.fileno()
    
    def write(self, message: str) -> int:
        import os
        if isinstance(message, str):
            data = message.encode('utf-8', errors='replace')
        else:
            data = bytes(message)
        return os.write(self._fd, data)
    
    def flush(self):
        pass  # os.write 是无缓冲的，不需要 flush


def setup_logger(level: str = "INFO", enable_file: bool = False, log_file: str | None = None):
    """
    配置 loguru 日志
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file: 是否启用文件日志
        log_file: 日志文件路径，如果为 None 则使用默认路径
    """
    # 先移除默认处理器
    logger.remove()
    
    # 创建无缓冲的输出流
    unbuffered_stderr = UnbufferedStderr()
    
    # 控制台输出 - 使用异步队列 + 无缓冲输出
    # enqueue=True: 启用异步队列，日志在后台线程中输出
    # 这样即使主线程被 GPU 操作阻塞，日志仍然可以实时输出
    logger.add(
        unbuffered_stderr,  # 直接写入 fd，绕过所有缓冲
        format=_format_record,
        level=level,
        colorize=False,  # 自定义 sink 不支持自动颜色处理
        backtrace=True,
        diagnose=True,
        enqueue=True,  # 启用异步队列，让日志在后台线程输出（不被 GPU 阻塞）
    )
    
    # 文件输出（无颜色）
    if enable_file:
        if log_file is None:
            log_file = Path("logs") / "whisperlm.log"
        else:
            log_file = Path(log_file)
        
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_file),
            format=_format_record_file,
            level=level,
            colorize=False,
            rotation="100 MB",
            retention="7 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,  # 启用异步队列，让日志在后台线程输出
        )
    
    return logger


# 默认配置
setup_logger(level="INFO", enable_file=False)

