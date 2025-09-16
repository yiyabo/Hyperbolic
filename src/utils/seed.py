"""
随机种子设置工具模块

该模块提供统一的随机种子设置功能，确保实验的可重现性。
包括对以下库的种子设置：
- Python内置random模块
- NumPy
- PyTorch (CPU和CUDA)
- 其他相关库

使用方法：
    from src.utils.seed import set_seed
    set_seed(42)
"""

import os
import random
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True, warn_only: bool = True):
    """
    设置所有相关库的随机种子以确保可重现性

    Args:
        seed: 随机种子值
        deterministic: 是否启用确定性算法（可能影响性能）
        warn_only: 当无法设置确定性时是否只警告而不报错
    """
    logger.info(f"Setting random seed to {seed}")

    # Python内置random模块
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况

        if deterministic:
            # 设置CUDA确定性行为
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # 设置环境变量以获得更好的确定性
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

            try:
                # PyTorch 1.8+
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
            except AttributeError:
                # 较老的PyTorch版本
                logger.warning("torch.use_deterministic_algorithms not available in this PyTorch version")
            except Exception as e:
                if warn_only:
                    logger.warning(f"Could not enable deterministic algorithms: {e}")
                else:
                    raise

    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)

    logger.info("Random seed set successfully")


def set_worker_seed(worker_id: int, base_seed: int = 42):
    """
    为DataLoader的worker进程设置随机种子

    Args:
        worker_id: worker进程ID
        base_seed: 基础种子值
    """
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_random_state():
    """
    获取当前随机数生成器的状态

    Returns:
        包含各库随机状态的字典
    """
    state = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state['torch_cuda_random'] = torch.cuda.get_rng_state()
        if torch.cuda.device_count() > 1:
            state['torch_cuda_random_all'] = torch.cuda.get_rng_state_all()

    return state


def set_random_state(state: dict):
    """
    恢复随机数生成器状态

    Args:
        state: 由get_random_state()返回的状态字典
    """
    if 'python_random' in state:
        random.setstate(state['python_random'])

    if 'numpy_random' in state:
        np.random.set_state(state['numpy_random'])

    if 'torch_random' in state:
        torch.set_rng_state(state['torch_random'])

    if torch.cuda.is_available():
        if 'torch_cuda_random' in state:
            torch.cuda.set_rng_state(state['torch_cuda_random'])

        if 'torch_cuda_random_all' in state and torch.cuda.device_count() > 1:
            torch.cuda.set_rng_state_all(state['torch_cuda_random_all'])


class SeedContext:
    """
    上下文管理器，用于临时设置随机种子

    使用方法:
        with SeedContext(42):
            # 在这个块中使用种子42
            result = some_random_operation()
        # 这里随机状态被恢复
    """

    def __init__(self, seed: int, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.saved_state = None
        self.saved_deterministic_state = None

    def __enter__(self):
        # 保存当前状态
        self.saved_state = get_random_state()

        if torch.cuda.is_available() and self.deterministic:
            self.saved_deterministic_state = {
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
            }

        # 设置新种子
        set_seed(self.seed, self.deterministic, warn_only=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复状态
        if self.saved_state:
            set_random_state(self.saved_state)

        if self.saved_deterministic_state and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = self.saved_deterministic_state['cudnn_deterministic']
            torch.backends.cudnn.benchmark = self.saved_deterministic_state['cudnn_benchmark']


def make_reproducible(func):
    """
    装饰器，使函数调用具有可重现性

    Args:
        func: 要装饰的函数，应该有一个seed参数

    Returns:
        装饰后的函数
    """
    def wrapper(*args, seed: Optional[int] = None, **kwargs):
        if seed is not None:
            with SeedContext(seed):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


def check_reproducibility(func, *args, seed: int = 42, num_runs: int = 3, **kwargs):
    """
    检查函数的可重现性

    Args:
        func: 要检查的函数
        *args: 函数参数
        seed: 使用的种子
        num_runs: 运行次数
        **kwargs: 函数关键字参数

    Returns:
        是否所有运行结果相同
    """
    results = []

    for i in range(num_runs):
        with SeedContext(seed):
            result = func(*args, **kwargs)
            results.append(result)

    # 检查所有结果是否相同
    first_result = results[0]

    for i, result in enumerate(results[1:], 1):
        if isinstance(result, torch.Tensor):
            if not torch.equal(first_result, result):
                logger.warning(f"Run {i+1} differs from run 1 (torch.Tensor)")
                return False
        elif isinstance(result, np.ndarray):
            if not np.array_equal(first_result, result):
                logger.warning(f"Run {i+1} differs from run 1 (numpy.ndarray)")
                return False
        else:
            if first_result != result:
                logger.warning(f"Run {i+1} differs from run 1")
                return False

    logger.info(f"Function is reproducible over {num_runs} runs with seed {seed}")
    return True


def generate_random_seeds(num_seeds: int, base_seed: int = 42) -> list:
    """
    生成一系列随机种子

    Args:
        num_seeds: 需要生成的种子数量
        base_seed: 基础种子

    Returns:
        种子列表
    """
    with SeedContext(base_seed):
        return [random.randint(0, 2**32 - 1) for _ in range(num_seeds)]


# 便捷函数
def seed_everything(seed: int = 42):
    """set_seed的别名，更简洁的名称"""
    set_seed(seed)


def reset_seeds():
    """重置所有随机数生成器（使用当前时间）"""
    import time
    current_time = int(time.time())
    set_seed(current_time)
    logger.info(f"Reset all seeds using current time: {current_time}")


if __name__ == "__main__":
    # 测试模块
    print("Testing seed utilities...")

    # 基本种子设置
    set_seed(42)
    print(f"Random number (Python): {random.random()}")
    print(f"Random number (NumPy): {np.random.random()}")
    print(f"Random tensor (PyTorch): {torch.rand(3)}")

    # 测试上下文管理器
    print("\nTesting SeedContext:")
    with SeedContext(123):
        print(f"In context - Random: {random.random()}")
    print(f"Outside context - Random: {random.random()}")

    # 测试可重现性
    print("\nTesting reproducibility:")
    def test_func():
        return torch.rand(5)

    is_reproducible = check_reproducibility(test_func, seed=42, num_runs=3)
    print(f"Function is reproducible: {is_reproducible}")

    print("Seed utilities test completed!")
