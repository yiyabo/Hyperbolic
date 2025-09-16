"""
日志工具模块

该模块提供统一的日志配置和管理功能，包括：
- 标准日志设置和格式化
- 训练进度和指标日志
- 文件和控制台输出
- 结构化日志记录
- 性能监控日志
- 与第三方工具集成

使用方法:
    from src.utils.logging import setup_logging, get_logger

    # 基本设置
    setup_logging(level="INFO", log_file="training.log")
    logger = get_logger(__name__)
    logger.info("Training started")

    # 结构化日志
    from src.utils.logging import MetricsLogger
    metrics_logger = MetricsLogger("metrics.jsonl")
    metrics_logger.log_metrics(epoch=1, loss=0.5, aupr=0.8)
"""

import os
import sys
import json
import time
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import threading
from collections import defaultdict, deque

# 可选依赖
try:
    import wandb
    WANDB_AVAILABLE = True
except (ImportError, Exception):
    # 处理wandb的各种导入问题
    WANDB_AVAILABLE = False

try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（用于控制台输出）"""

    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
        'RESET': '\033[0m'       # 重置
    }

    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, '')}{log_message}{self.COLORS['RESET']}"


class StructuredFormatter(logging.Formatter):
    """结构化JSON日志格式化器"""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # 添加自定义字段
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False)


class MetricsLogger:
    """专用于训练和评估指标的日志记录器"""

    def __init__(self, log_file: str, buffer_size: int = 100):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.buffer = []
        self.lock = threading.Lock()

        # 创建或打开日志文件
        self.file_handle = open(self.log_file, 'a', encoding='utf-8')

    def log_metrics(self, **kwargs):
        """记录指标数据"""
        with self.lock:
            entry = {
                'timestamp': datetime.utcnow().isoformat(),
                **kwargs
            }

            self.buffer.append(entry)

            # 当缓冲区满时写入文件
            if len(self.buffer) >= self.buffer_size:
                self.flush()

    def flush(self):
        """强制写入缓冲区中的所有条目"""
        with self.lock:
            for entry in self.buffer:
                self.file_handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
            self.file_handle.flush()
            self.buffer.clear()

    def close(self):
        """关闭日志文件"""
        self.flush()
        self.file_handle.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass


class TrainingLogger:
    """训练过程专用日志记录器"""

    def __init__(self, name: str = "training", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 获取基础logger
        self.logger = logging.getLogger(name)

        # 指标记录器
        self.metrics_logger = MetricsLogger(self.log_dir / "metrics.jsonl")

        # 统计信息
        self.epoch_stats = defaultdict(list)
        self.batch_stats = defaultdict(deque)
        self.start_time = time.time()
        self.epoch_start_time = None

        # 第三方工具
        self.wandb_enabled = False
        self.tensorboard_writer = None

    def setup_wandb(self, project_name: str, config: Dict = None, **kwargs):
        """设置 Weights & Biases 记录"""
        if not WANDB_AVAILABLE:
            self.logger.warning("wandb not available, skipping setup")
            return

        try:
            wandb.init(project=project_name, config=config, **kwargs)
            self.wandb_enabled = True
            self.logger.info("Weights & Biases initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize wandb: {e}")

    def setup_tensorboard(self, log_dir: str = None):
        """设置 TensorBoard 记录"""
        if not TENSORBOARD_AVAILABLE:
            self.logger.warning("tensorboard not available, skipping setup")
            return

        try:
            tb_dir = log_dir or str(self.log_dir / "tensorboard")
            self.tensorboard_writer = SummaryWriter(tb_dir)
            self.logger.info(f"TensorBoard writer initialized: {tb_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize tensorboard: {e}")

    def log_epoch_start(self, epoch: int, **kwargs):
        """记录epoch开始"""
        self.epoch_start_time = time.time()
        self.logger.info(f"Epoch {epoch} started")

        # 记录到指标日志
        self.metrics_logger.log_metrics(
            event="epoch_start",
            epoch=epoch,
            **kwargs
        )

    def log_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """记录epoch结束"""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        total_time = time.time() - self.start_time

        # 记录到基础日志
        metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s - {metrics_str}")

        # 记录到指标日志
        self.metrics_logger.log_metrics(
            event="epoch_end",
            epoch=epoch,
            epoch_time=epoch_time,
            total_time=total_time,
            **metrics,
            **kwargs
        )

        # 记录到统计
        for key, value in metrics.items():
            self.epoch_stats[key].append(value)

        # 第三方工具记录
        if self.wandb_enabled:
            try:
                wandb.log({**metrics, "epoch": epoch, "epoch_time": epoch_time})
            except Exception as e:
                self.logger.warning(f"Failed to log to wandb: {e}")

        if self.tensorboard_writer:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(key, value, epoch)
                self.tensorboard_writer.add_scalar("epoch_time", epoch_time, epoch)
            except Exception as e:
                self.logger.warning(f"Failed to log to tensorboard: {e}")

    def log_batch_metrics(self, step: int, metrics: Dict[str, float], **kwargs):
        """记录批次指标"""
        # 记录到指标日志（每隔一定步数）
        if step % 10 == 0:  # 减少频繁写入
            self.metrics_logger.log_metrics(
                event="batch_metrics",
                step=step,
                **metrics,
                **kwargs
            )

        # 更新统计（保持最近N个值）
        for key, value in metrics.items():
            if len(self.batch_stats[key]) >= 1000:
                self.batch_stats[key].popleft()
            self.batch_stats[key].append(value)

        # TensorBoard记录
        if self.tensorboard_writer and step % 10 == 0:
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(f"batch/{key}", value, step)
            except Exception as e:
                self.logger.warning(f"Failed to log batch metrics to tensorboard: {e}")

    def log_model_stats(self, model, step: Optional[int] = None):
        """记录模型统计信息"""
        try:
            if hasattr(model, 'get_statistics'):
                stats = model.get_statistics()

                self.logger.debug(f"Model statistics: {stats}")

                # 记录到指标日志
                self.metrics_logger.log_metrics(
                    event="model_stats",
                    step=step,
                    **stats
                )

                # 记录到第三方工具
                if self.wandb_enabled:
                    wandb.log({f"model/{k}": v for k, v in stats.items()})

                if self.tensorboard_writer and step is not None:
                    for key, value in stats.items():
                        if isinstance(value, (int, float)):
                            self.tensorboard_writer.add_scalar(f"model/{key}", value, step)

        except Exception as e:
            self.logger.warning(f"Failed to log model stats: {e}")

    def log_hyperparameters(self, config: Dict[str, Any]):
        """记录超参数"""
        self.logger.info("Hyperparameters:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

        # 记录到指标日志
        self.metrics_logger.log_metrics(
            event="hyperparameters",
            **config
        )

        # 第三方工具
        if self.wandb_enabled:
            try:
                wandb.config.update(config)
            except Exception as e:
                self.logger.warning(f"Failed to log hyperparameters to wandb: {e}")

    def get_epoch_summary(self) -> Dict[str, Any]:
        """获取epoch统计摘要"""
        summary = {}
        for key, values in self.epoch_stats.items():
            if values:
                summary[f"{key}_mean"] = sum(values) / len(values)
                summary[f"{key}_last"] = values[-1]
                if len(values) > 1:
                    summary[f"{key}_std"] = (
                        sum((x - summary[f"{key}_mean"]) ** 2 for x in values) / len(values)
                    ) ** 0.5
        return summary

    def close(self):
        """关闭日志记录器"""
        self.metrics_logger.close()
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        if self.wandb_enabled:
            try:
                wandb.finish()
            except:
                pass


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    format_string: Optional[str] = None,
    use_colors: bool = True,
    structured: bool = False,
    max_file_size: str = "50MB",
    backup_count: int = 3
):
    """
    设置项目级别的日志配置

    Args:
        level: 日志级别
        log_file: 日志文件名（如果为None则不写文件）
        log_dir: 日志目录
        format_string: 自定义格式字符串
        use_colors: 控制台输出是否使用颜色
        structured: 是否使用结构化JSON格式
        max_file_size: 单个日志文件最大大小
        backup_count: 日志文件备份数量
    """
    # 确保日志目录存在
    log_directory = Path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)

    # 获取根logger
    root_logger = logging.getLogger()

    # 设置级别
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    root_logger.setLevel(level)

    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 默认格式
    if format_string is None:
        if structured:
            format_string = None  # JSON格式化器会处理
        else:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if structured:
        console_formatter = StructuredFormatter()
    elif use_colors and sys.stdout.isatty():
        console_formatter = ColoredFormatter(format_string)
    else:
        console_formatter = logging.Formatter(format_string)

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = log_directory / log_file

        # 解析文件大小
        size_multipliers = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        max_bytes = 50 * 1024 * 1024  # 默认50MB

        if isinstance(max_file_size, str):
            for suffix, multiplier in size_multipliers.items():
                if max_file_size.upper().endswith(suffix):
                    max_bytes = int(max_file_size[:-2]) * multiplier
                    break

        # 使用轮转文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)

        if structured:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(format_string)

        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # 设置第三方库的日志级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    root_logger.info(f"Logging setup completed - Level: {logging.getLevelName(level)}")
    if log_file:
        root_logger.info(f"Log file: {log_directory / log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    获取命名的日志记录器

    Args:
        name: 日志记录器名称，通常使用 __name__

    Returns:
        配置好的日志记录器
    """
    return logging.getLogger(name)


def log_function_call(func):
    """
    装饰器：记录函数调用

    Args:
        func: 要装饰的函数

    Returns:
        装饰后的函数
    """
    logger = get_logger(func.__module__)

    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()

            logger.debug(f"{func.__name__} completed in {end_time - start_time:.4f}s")
            return result

        except Exception as e:
            end_time = time.time()
            logger.error(f"{func.__name__} failed after {end_time - start_time:.4f}s: {e}")
            raise

    return wrapper


def log_exception(logger: logging.Logger, msg: str = "Exception occurred"):
    """
    记录异常信息的便捷函数

    Args:
        logger: 日志记录器
        msg: 错误消息
    """
    logger.error(msg, exc_info=True)


# 便捷函数
def setup_basic_logging(level: str = "INFO", log_file: Optional[str] = "app.log"):
    """基础日志设置的便捷函数"""
    setup_logging(level=level, log_file=log_file)


def create_training_logger(name: str = "training", log_dir: str = "logs") -> TrainingLogger:
    """创建训练日志记录器的便捷函数"""
    return TrainingLogger(name=name, log_dir=log_dir)


if __name__ == "__main__":
    # 测试日志功能
    print("Testing logging utilities...")

    # 基础设置
    setup_logging(level="DEBUG", log_file="test.log", use_colors=True)
    logger = get_logger(__name__)

    # 基础日志测试
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # 训练日志器测试
    training_logger = TrainingLogger("test_training", "test_logs")
    training_logger.log_hyperparameters({"lr": 0.001, "batch_size": 32})

    training_logger.log_epoch_start(1)
    training_logger.log_batch_metrics(1, {"loss": 0.5, "accuracy": 0.8})
    training_logger.log_epoch_end(1, {"loss": 0.4, "accuracy": 0.85, "val_loss": 0.45})

    training_logger.close()

    # 指标日志器测试
    metrics_logger = MetricsLogger("test_metrics.jsonl")
    metrics_logger.log_metrics(epoch=1, loss=0.5, aupr=0.8, auroc=0.9)
    metrics_logger.close()

    print("Logging utilities test completed!")
