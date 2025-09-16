"""
PPI-HGCN: 基于双曲几何的蛋白质相互作用预测

该包实现了使用双曲图卷积网络进行蛋白质相互作用预测的完整流程，包括：
- 数据加载和预处理 (dataio)
- 双曲几何和流形操作 (geometry)
- 模型定义和实现 (models)
- 训练流程 (train)
- 评估指标和方法 (eval)
- 工具函数 (utils)
"""

__version__ = "0.1.0"
__author__ = "PPI-HGCN Team"
__email__ = "contact@ppi-hgcn.org"

# 版本信息
VERSION_INFO = {
    'major': 0,
    'minor': 1,
    'patch': 0,
    'stage': 'alpha'
}

def get_version():
    """获取版本字符串"""
    return f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"

# 导入主要组件
from . import dataio
from . import geometry
from . import models
from . import train
from . import eval
from . import utils

__all__ = [
    'dataio',
    'geometry',
    'models',
    'train',
    'eval',
    'utils',
    'get_version',
    '__version__'
]
