"""
数据输入输出模块

该模块提供数据加载、预处理和ID映射的功能，包括：
- HuRI数据加载 (load_huri)
- STRING数据加载 (load_string)
- UniProt ID映射 (id_mapping_uniprot)
- 负采样策略 (neg_sampling)
"""

from .load_huri import HuRILoader, load_huri_data
from .load_string import STRINGLoader, load_string_data
from .id_mapping_uniprot import UniProtIDMapper, create_id_mapping
from .neg_sampling import NegativeSampler, AdaptiveNegativeSampler, create_negative_sampler

__all__ = [
    # HuRI数据加载
    'HuRILoader',
    'load_huri_data',

    # STRING数据加载
    'STRINGLoader',
    'load_string_data',

    # ID映射
    'UniProtIDMapper',
    'create_id_mapping',

    # 负采样
    'NegativeSampler',
    'AdaptiveNegativeSampler',
    'create_negative_sampler'
]

# 支持的数据源
SUPPORTED_DATASETS = {
    'huri': {
        'description': 'Human Reference Interactome',
        'id_type': 'ensembl_gene',
        'loader_class': HuRILoader
    },
    'string_human': {
        'description': 'STRING v12.0 Human',
        'id_type': 'string_protein',
        'loader_class': STRINGLoader,
        'taxid': 9606
    },
    'string_mouse': {
        'description': 'STRING v12.0 Mouse',
        'id_type': 'string_protein',
        'loader_class': STRINGLoader,
        'taxid': 10090
    },
    'string_yeast': {
        'description': 'STRING v12.0 Yeast',
        'id_type': 'string_protein',
        'loader_class': STRINGLoader,
        'taxid': 559292
    }
}

# 支持的负采样策略
NEGATIVE_SAMPLING_STRATEGIES = {
    'uniform': 'Uniform random negative sampling',
    'topology_driven': 'Mixed hard and easy negative sampling',
    'degree_similar': 'Degree similarity-based negative sampling',
    'common_neighbor': 'Common neighbor-based negative sampling',
    'community_based': 'Community structure-based negative sampling'
}

def get_dataset_info(dataset_name: str) -> dict:
    """
    获取数据集信息

    Args:
        dataset_name: 数据集名称

    Returns:
        dict: 数据集信息
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported datasets: {list(SUPPORTED_DATASETS.keys())}")

    return SUPPORTED_DATASETS[dataset_name]

def list_supported_datasets() -> list:
    """
    列出所有支持的数据集

    Returns:
        list: 支持的数据集名称列表
    """
    return list(SUPPORTED_DATASETS.keys())

def list_negative_sampling_strategies() -> list:
    """
    列出所有支持的负采样策略

    Returns:
        list: 支持的负采样策略列表
    """
    return list(NEGATIVE_SAMPLING_STRATEGIES.keys())
