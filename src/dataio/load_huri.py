"""
HuRI (Human Reference Interactome) 数据加载模块

该模块负责加载和预处理HuRI数据，包括：
- 读取HuRI TSV文件
- 去除自环和重复边
- ID映射（Ensembl -> UniProt）
- 构建图数据结构
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class HuRILoader:
    """HuRI数据加载器"""

    def __init__(self, config: Dict):
        """
        初始化HuRI数据加载器

        Args:
            config: 配置字典，包含数据路径和预处理参数
        """
        self.config = config
        self.raw_interactions_path = config['raw_data']['interactions']
        self.id_mapping_path = config['id_mapping']['mapping_file']
        self.remove_self_loops = config['preprocessing']['remove_self_loops']
        self.remove_duplicates = config['preprocessing']['remove_duplicates']
        self.min_degree = config['preprocessing']['min_degree']

        # 存储处理后的数据
        self.interactions_df = None
        self.id_mapping = None
        self.protein_to_idx = None
        self.idx_to_protein = None
        self.graph_data = None

    def load_interactions(self) -> pd.DataFrame:
        """
        加载HuRI相互作用数据

        Returns:
            pd.DataFrame: 相互作用数据框
        """
        logger.info(f"Loading HuRI interactions from {self.raw_interactions_path}")

        try:
            # 读取TSV文件，假设列名为protein1, protein2
            df = pd.read_csv(self.raw_interactions_path, sep='\t')

            # 检查必需的列
            required_columns = ['protein1', 'protein2']
            if not all(col in df.columns for col in required_columns):
                # 如果列名不同，尝试其他可能的列名
                possible_names = [
                    ['Ensembl_gene_1', 'Ensembl_gene_2'],
                    ['Gene_A', 'Gene_B'],
                    ['protein_A', 'protein_B'],
                    ['InteractorA', 'InteractorB']
                ]

                for names in possible_names:
                    if all(name in df.columns for name in names):
                        df = df.rename(columns={names[0]: 'protein1', names[1]: 'protein2'})
                        break
                else:
                    raise ValueError(f"Cannot find interaction columns in {df.columns}")

            logger.info(f"Loaded {len(df)} raw interactions")
            return df

        except Exception as e:
            logger.error(f"Error loading HuRI interactions: {e}")
            raise

    def load_id_mapping(self) -> Dict[str, str]:
        """
        加载ID映射文件（Ensembl Gene ID -> UniProt ID）

        Returns:
            Dict[str, str]: ID映射字典
        """
        logger.info(f"Loading ID mapping from {self.id_mapping_path}")

        try:
            if Path(self.id_mapping_path).suffix == '.json':
                with open(self.id_mapping_path, 'r') as f:
                    mapping = json.load(f)
            else:
                # 假设是TSV格式
                mapping_df = pd.read_csv(self.id_mapping_path, sep='\t')

                # 检查列名
                if 'ensembl_gene' in mapping_df.columns and 'uniprotkb' in mapping_df.columns:
                    mapping = dict(zip(mapping_df['ensembl_gene'], mapping_df['uniprotkb']))
                elif len(mapping_df.columns) >= 2:
                    mapping = dict(zip(mapping_df.iloc[:, 0], mapping_df.iloc[:, 1]))
                else:
                    raise ValueError("Invalid mapping file format")

            logger.info(f"Loaded {len(mapping)} ID mappings")
            return mapping

        except Exception as e:
            logger.error(f"Error loading ID mapping: {e}")
            # 如果映射文件不存在，返回空字典（使用原始ID）
            logger.warning("Using original IDs without mapping")
            return {}

    def preprocess_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理相互作用数据

        Args:
            df: 原始相互作用数据框

        Returns:
            pd.DataFrame: 预处理后的数据框
        """
        logger.info("Preprocessing interactions...")

        # 去除空值
        df = df.dropna(subset=['protein1', 'protein2'])

        # 去除自环
        if self.remove_self_loops:
            before_len = len(df)
            df = df[df['protein1'] != df['protein2']]
            logger.info(f"Removed {before_len - len(df)} self-loops")

        # 应用ID映射
        if self.id_mapping:
            logger.info("Applying ID mapping...")
            df['protein1_mapped'] = df['protein1'].map(self.id_mapping)
            df['protein2_mapped'] = df['protein2'].map(self.id_mapping)

            # 只保留成功映射的相互作用
            mapped_mask = df['protein1_mapped'].notna() & df['protein2_mapped'].notna()
            unmapped_count = len(df) - mapped_mask.sum()
            if unmapped_count > 0:
                logger.warning(f"Could not map {unmapped_count} interactions")

            df = df[mapped_mask].copy()
            df['protein1'] = df['protein1_mapped']
            df['protein2'] = df['protein2_mapped']
            df = df.drop(['protein1_mapped', 'protein2_mapped'], axis=1)

        # 标准化边（确保protein1 < protein2，避免重复）
        df['protein_min'] = df[['protein1', 'protein2']].min(axis=1)
        df['protein_max'] = df[['protein1', 'protein2']].max(axis=1)
        df = df.drop(['protein1', 'protein2'], axis=1)
        df = df.rename(columns={'protein_min': 'protein1', 'protein_max': 'protein2'})

        # 去除重复边
        if self.remove_duplicates:
            before_len = len(df)
            df = df.drop_duplicates(subset=['protein1', 'protein2'])
            logger.info(f"Removed {before_len - len(df)} duplicate interactions")

        logger.info(f"Final dataset contains {len(df)} interactions")
        return df

    def build_protein_mapping(self, df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        构建蛋白质到索引的映射

        Args:
            df: 相互作用数据框

        Returns:
            Tuple[Dict[str, int], Dict[int, str]]: 蛋白质到索引和索引到蛋白质的映射
        """
        # 获取所有唯一蛋白质
        all_proteins = set(df['protein1'].unique()) | set(df['protein2'].unique())
        all_proteins = sorted(list(all_proteins))

        # 创建映射
        protein_to_idx = {protein: idx for idx, protein in enumerate(all_proteins)}
        idx_to_protein = {idx: protein for protein, idx in protein_to_idx.items()}

        logger.info(f"Created mapping for {len(all_proteins)} unique proteins")
        return protein_to_idx, idx_to_protein

    def filter_by_degree(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据最小度数过滤蛋白质

        Args:
            df: 相互作用数据框

        Returns:
            pd.DataFrame: 过滤后的数据框
        """
        if self.min_degree <= 1:
            return df

        logger.info(f"Filtering proteins with degree >= {self.min_degree}")

        # 计算每个蛋白质的度数
        protein_counts = pd.concat([df['protein1'], df['protein2']]).value_counts()
        valid_proteins = set(protein_counts[protein_counts >= self.min_degree].index)

        # 过滤相互作用
        before_len = len(df)
        df = df[
            df['protein1'].isin(valid_proteins) &
            df['protein2'].isin(valid_proteins)
        ].copy()

        logger.info(f"Filtered out {before_len - len(df)} interactions due to low degree")
        logger.info(f"Remaining {len(valid_proteins)} proteins with degree >= {self.min_degree}")

        return df

    def create_graph_data(self, df: pd.DataFrame) -> Data:
        """
        创建PyTorch Geometric图数据对象

        Args:
            df: 预处理后的相互作用数据框

        Returns:
            Data: PyTorch Geometric图数据对象
        """
        logger.info("Creating PyTorch Geometric graph data...")

        # 构建边索引
        edge_index_list = []
        for _, row in df.iterrows():
            idx1 = self.protein_to_idx[row['protein1']]
            idx2 = self.protein_to_idx[row['protein2']]
            # 添加双向边（无向图）
            edge_index_list.extend([[idx1, idx2], [idx2, idx1]])

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        # 创建图数据对象
        num_nodes = len(self.protein_to_idx)
        graph_data = Data(
            edge_index=edge_index,
            num_nodes=num_nodes,
            # 节点特征稍后添加
            x=None
        )

        # 添加元数据
        graph_data.protein_to_idx = self.protein_to_idx
        graph_data.idx_to_protein = self.idx_to_protein
        graph_data.num_edges = edge_index.size(1)
        graph_data.num_proteins = num_nodes

        logger.info(f"Created graph with {num_nodes} nodes and {edge_index.size(1)} edges")
        return graph_data

    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息

        Returns:
            Dict: 统计信息
        """
        if self.graph_data is None:
            return {}

        # 计算度数统计
        degrees = torch.bincount(self.graph_data.edge_index[0])

        stats = {
            'num_proteins': self.graph_data.num_proteins,
            'num_interactions': self.graph_data.num_edges // 2,  # 无向图
            'num_edges': self.graph_data.num_edges,
            'avg_degree': float(degrees.float().mean()),
            'max_degree': int(degrees.max()),
            'min_degree': int(degrees.min()),
            'degree_std': float(degrees.float().std())
        }

        return stats

    def load_and_process(self) -> Data:
        """
        加载并处理HuRI数据的主要流程

        Returns:
            Data: 处理后的图数据对象
        """
        logger.info("Starting HuRI data loading and processing...")

        # 1. 加载原始数据
        self.interactions_df = self.load_interactions()

        # 2. 加载ID映射
        self.id_mapping = self.load_id_mapping()

        # 3. 预处理相互作用数据
        self.interactions_df = self.preprocess_interactions(self.interactions_df)

        # 4. 根据度数过滤
        self.interactions_df = self.filter_by_degree(self.interactions_df)

        # 5. 构建蛋白质映射
        self.protein_to_idx, self.idx_to_protein = self.build_protein_mapping(self.interactions_df)

        # 6. 创建图数据对象
        self.graph_data = self.create_graph_data(self.interactions_df)

        # 7. 输出统计信息
        stats = self.get_statistics()
        logger.info("HuRI data statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        return self.graph_data

    def save_processed_data(self, output_path: str, node_mapping_path: str = None):
        """
        保存处理后的数据

        Args:
            output_path: 图数据保存路径
            node_mapping_path: 节点映射保存路径
        """
        if self.graph_data is None:
            raise ValueError("No processed data to save. Call load_and_process() first.")

        logger.info(f"Saving processed graph data to {output_path}")
        torch.save(self.graph_data, output_path)

        if node_mapping_path:
            logger.info(f"Saving node mapping to {node_mapping_path}")
            mapping_data = {
                'protein_to_idx': self.protein_to_idx,
                'idx_to_protein': self.idx_to_protein,
                'statistics': self.get_statistics()
            }
            with open(node_mapping_path, 'w') as f:
                json.dump(mapping_data, f, indent=2)


def load_huri_data(config_path: str) -> Data:
    """
    便捷函数：从配置文件加载HuRI数据

    Args:
        config_path: 配置文件路径

    Returns:
        Data: 图数据对象
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    loader = HuRILoader(config)
    return loader.load_and_process()


if __name__ == "__main__":
    # 示例用法
    import yaml
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "cfg/data_huri.yaml"

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 加载数据
    graph_data = load_huri_data(config_path)

    print("\nHuRI data loaded successfully!")
    print(f"Graph: {graph_data}")
