"""
STRING数据加载模块

该模块负责加载和预处理STRING数据，包括：
- 读取STRING链接文件
- 按物种和置信度过滤
- ID映射（STRING protein ID -> UniProt ID）
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
import gzip

logger = logging.getLogger(__name__)


class STRINGLoader:
    """STRING数据加载器"""

    def __init__(self, config: Dict):
        """
        初始化STRING数据加载器

        Args:
            config: 配置字典，包含数据路径和预处理参数
        """
        self.config = config
        self.species_taxid = config['species']['taxid']
        self.species_name = config['species']['common_name']

        # 文件路径
        self.interactions_path = config['raw_data']['interactions']
        self.protein_info_path = config['raw_data']['protein_info']
        self.protein_aliases_path = config['raw_data']['protein_aliases']
        self.id_mapping_path = config['id_mapping']['mapping_file']

        # 预处理参数
        self.remove_self_loops = config['preprocessing']['remove_self_loops']
        self.remove_duplicates = config['preprocessing']['remove_duplicates']
        self.min_degree = config['preprocessing']['min_degree']
        self.confidence_thresholds = config['preprocessing']['confidence_thresholds']

        # STRING特定配置
        self.interaction_types = config['string_config']['interaction_types']
        self.score_type = config['string_config']['score_type']
        self.min_confidence = config['quality_control']['min_confidence']
        self.max_degree = config['quality_control'].get('max_degree', float('inf'))

        # 存储处理后的数据
        self.interactions_df = None
        self.protein_info_df = None
        self.protein_aliases_df = None
        self.id_mapping = None
        self.protein_to_idx = None
        self.idx_to_protein = None
        self.graph_data = None

    def load_interactions(self) -> pd.DataFrame:
        """
        加载STRING相互作用数据

        Returns:
            pd.DataFrame: 相互作用数据框
        """
        logger.info(f"Loading STRING interactions for {self.species_name} from {self.interactions_path}")

        try:
            # 处理压缩文件
            if self.interactions_path.endswith('.gz'):
                df = pd.read_csv(self.interactions_path, sep=' ', compression='gzip')
            else:
                df = pd.read_csv(self.interactions_path, sep=' ')

            # 检查必需的列
            expected_columns = ['protein1', 'protein2', 'combined_score']
            if not all(col in df.columns for col in expected_columns):
                raise ValueError(f"Missing required columns. Expected: {expected_columns}, Found: {df.columns.tolist()}")

            logger.info(f"Loaded {len(df)} raw interactions")

            # 显示分数分布
            score_stats = df['combined_score'].describe()
            logger.info(f"Combined score statistics:\n{score_stats}")

            return df

        except Exception as e:
            logger.error(f"Error loading STRING interactions: {e}")
            raise

    def load_protein_info(self) -> pd.DataFrame:
        """
        加载STRING蛋白质信息

        Returns:
            pd.DataFrame: 蛋白质信息数据框
        """
        logger.info(f"Loading protein info from {self.protein_info_path}")

        try:
            if self.protein_info_path.endswith('.gz'):
                df = pd.read_csv(self.protein_info_path, sep='\t', compression='gzip')
            else:
                df = pd.read_csv(self.protein_info_path, sep='\t')

            logger.info(f"Loaded info for {len(df)} proteins")
            return df

        except Exception as e:
            logger.error(f"Error loading protein info: {e}")
            return pd.DataFrame()

    def load_protein_aliases(self) -> pd.DataFrame:
        """
        加载STRING蛋白质别名

        Returns:
            pd.DataFrame: 蛋白质别名数据框
        """
        logger.info(f"Loading protein aliases from {self.protein_aliases_path}")

        try:
            if self.protein_aliases_path.endswith('.gz'):
                df = pd.read_csv(self.protein_aliases_path, sep='\t', compression='gzip')
            else:
                df = pd.read_csv(self.protein_aliases_path, sep='\t')

            logger.info(f"Loaded {len(df)} protein aliases")
            return df

        except Exception as e:
            logger.error(f"Error loading protein aliases: {e}")
            return pd.DataFrame()

    def load_id_mapping(self) -> Dict[str, str]:
        """
        加载ID映射文件（STRING protein ID -> UniProt ID）

        Returns:
            Dict[str, str]: ID映射字典
        """
        logger.info(f"Loading ID mapping from {self.id_mapping_path}")

        try:
            if Path(self.id_mapping_path).exists():
                if Path(self.id_mapping_path).suffix == '.json':
                    with open(self.id_mapping_path, 'r') as f:
                        mapping = json.load(f)
                else:
                    mapping_df = pd.read_csv(self.id_mapping_path, sep='\t')
                    # 假设第一列是STRING ID，第二列是UniProt ID
                    mapping = dict(zip(mapping_df.iloc[:, 0], mapping_df.iloc[:, 1]))

                logger.info(f"Loaded {len(mapping)} ID mappings")
                return mapping
            else:
                logger.warning("ID mapping file not found. Will try to create mapping from aliases.")
                return self.create_mapping_from_aliases()

        except Exception as e:
            logger.error(f"Error loading ID mapping: {e}")
            return {}

    def create_mapping_from_aliases(self) -> Dict[str, str]:
        """
        从蛋白质别名创建ID映射

        Returns:
            Dict[str, str]: ID映射字典
        """
        if self.protein_aliases_df is None or self.protein_aliases_df.empty:
            logger.warning("No protein aliases available for mapping")
            return {}

        logger.info("Creating ID mapping from protein aliases...")

        # 查找UniProt相关的别名
        uniprot_aliases = self.protein_aliases_df[
            self.protein_aliases_df['source'].str.contains('UniProt', case=False, na=False)
        ]

        if uniprot_aliases.empty:
            logger.warning("No UniProt aliases found")
            return {}

        # 创建映射（STRING protein ID -> UniProt ID）
        mapping = dict(zip(uniprot_aliases['string_protein_id'], uniprot_aliases['alias']))

        logger.info(f"Created {len(mapping)} mappings from aliases")
        return mapping

    def filter_by_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据置信度过滤相互作用

        Args:
            df: 原始相互作用数据框

        Returns:
            pd.DataFrame: 过滤后的数据框
        """
        logger.info(f"Filtering by confidence >= {self.min_confidence}")

        before_len = len(df)
        df = df[df['combined_score'] >= self.min_confidence].copy()

        logger.info(f"Filtered {before_len - len(df)} low-confidence interactions")
        logger.info(f"Remaining {len(df)} interactions with score >= {self.min_confidence}")

        return df

    def preprocess_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理相互作用数据

        Args:
            df: 原始相互作用数据框

        Returns:
            pd.DataFrame: 预处理后的数据框
        """
        logger.info("Preprocessing STRING interactions...")

        # 1. 按置信度过滤
        df = self.filter_by_confidence(df)

        # 2. 去除自环
        if self.remove_self_loops:
            before_len = len(df)
            df = df[df['protein1'] != df['protein2']]
            logger.info(f"Removed {before_len - len(df)} self-loops")

        # 3. 应用ID映射
        if self.id_mapping:
            logger.info("Applying ID mapping...")
            df['protein1_mapped'] = df['protein1'].map(self.id_mapping)
            df['protein2_mapped'] = df['protein2'].map(self.id_mapping)

            # 统计映射成功率
            mapped1 = df['protein1_mapped'].notna().sum()
            mapped2 = df['protein2_mapped'].notna().sum()
            total_proteins = len(set(df['protein1']) | set(df['protein2']))

            logger.info(f"Mapping success rate: protein1={mapped1}/{len(df)}, protein2={mapped2}/{len(df)}")

            # 只保留成功映射的相互作用
            mapped_mask = df['protein1_mapped'].notna() & df['protein2_mapped'].notna()
            unmapped_count = len(df) - mapped_mask.sum()
            if unmapped_count > 0:
                logger.warning(f"Could not map {unmapped_count} interactions")

            df = df[mapped_mask].copy()
            df['protein1'] = df['protein1_mapped']
            df['protein2'] = df['protein2_mapped']
            df = df.drop(['protein1_mapped', 'protein2_mapped'], axis=1)

        # 4. 标准化边（确保protein1 < protein2）
        df['protein_min'] = df[['protein1', 'protein2']].min(axis=1)
        df['protein_max'] = df[['protein1', 'protein2']].max(axis=1)
        df = df.drop(['protein1', 'protein2'], axis=1)
        df = df.rename(columns={'protein_min': 'protein1', 'protein_max': 'protein2'})

        # 5. 去除重复边（保留最高分数的）
        if self.remove_duplicates:
            before_len = len(df)
            df = df.sort_values('combined_score', ascending=False)
            df = df.drop_duplicates(subset=['protein1', 'protein2'], keep='first')
            logger.info(f"Removed {before_len - len(df)} duplicate interactions")

        logger.info(f"Final dataset contains {len(df)} interactions")
        return df

    def filter_by_degree(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据度数过滤蛋白质

        Args:
            df: 相互作用数据框

        Returns:
            pd.DataFrame: 过滤后的数据框
        """
        if self.min_degree <= 1 and self.max_degree == float('inf'):
            return df

        logger.info(f"Filtering proteins with degree in range [{self.min_degree}, {self.max_degree}]")

        # 计算每个蛋白质的度数
        protein_counts = pd.concat([df['protein1'], df['protein2']]).value_counts()

        # 应用度数过滤
        valid_proteins = set(protein_counts[
            (protein_counts >= self.min_degree) &
            (protein_counts <= self.max_degree)
        ].index)

        # 过滤相互作用
        before_len = len(df)
        df = df[
            df['protein1'].isin(valid_proteins) &
            df['protein2'].isin(valid_proteins)
        ].copy()

        logger.info(f"Filtered out {before_len - len(df)} interactions due to degree constraints")
        logger.info(f"Remaining {len(valid_proteins)} proteins with valid degree")

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

    def create_graph_data(self, df: pd.DataFrame) -> Data:
        """
        创建PyTorch Geometric图数据对象

        Args:
            df: 预处理后的相互作用数据框

        Returns:
            Data: PyTorch Geometric图数据对象
        """
        logger.info("Creating PyTorch Geometric graph data...")

        # 构建边索引和边权重
        edge_index_list = []
        edge_attr_list = []

        for _, row in df.iterrows():
            idx1 = self.protein_to_idx[row['protein1']]
            idx2 = self.protein_to_idx[row['protein2']]
            score = row['combined_score'] / 1000.0  # 标准化到[0, 1]

            # 添加双向边（无向图）
            edge_index_list.extend([[idx1, idx2], [idx2, idx1]])
            edge_attr_list.extend([score, score])

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

        # 创建图数据对象
        num_nodes = len(self.protein_to_idx)
        graph_data = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
            x=None  # 节点特征稍后添加
        )

        # 添加元数据
        graph_data.protein_to_idx = self.protein_to_idx
        graph_data.idx_to_protein = self.idx_to_protein
        graph_data.num_edges = edge_index.size(1)
        graph_data.num_proteins = num_nodes
        graph_data.species = self.species_name
        graph_data.taxid = self.species_taxid

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

        # 边权重统计
        edge_weights = self.graph_data.edge_attr

        stats = {
            'species': self.species_name,
            'taxid': self.species_taxid,
            'num_proteins': self.graph_data.num_proteins,
            'num_interactions': self.graph_data.num_edges // 2,
            'num_edges': self.graph_data.num_edges,
            'avg_degree': float(degrees.float().mean()),
            'max_degree': int(degrees.max()),
            'min_degree': int(degrees.min()),
            'degree_std': float(degrees.float().std()),
            'avg_confidence': float(edge_weights.mean()),
            'min_confidence': float(edge_weights.min()),
            'max_confidence': float(edge_weights.max()),
            'confidence_std': float(edge_weights.std())
        }

        return stats

    def load_and_process(self) -> Data:
        """
        加载并处理STRING数据的主要流程

        Returns:
            Data: 处理后的图数据对象
        """
        logger.info(f"Starting STRING data loading and processing for {self.species_name}...")

        # 1. 加载原始数据
        self.interactions_df = self.load_interactions()
        self.protein_info_df = self.load_protein_info()
        self.protein_aliases_df = self.load_protein_aliases()

        # 2. 加载或创建ID映射
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
        logger.info(f"STRING {self.species_name} data statistics:")
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

    def create_confidence_subgraphs(self) -> Dict[int, Data]:
        """
        为不同置信度阈值创建子图

        Returns:
            Dict[int, Data]: 置信度阈值到图数据的映射
        """
        if self.interactions_df is None:
            raise ValueError("No processed data available. Call load_and_process() first.")

        subgraphs = {}

        for threshold in self.confidence_thresholds:
            logger.info(f"Creating subgraph for confidence >= {threshold}")

            # 过滤相互作用
            filtered_df = self.interactions_df[
                self.interactions_df['combined_score'] >= threshold
            ].copy()

            if filtered_df.empty:
                logger.warning(f"No interactions found for confidence >= {threshold}")
                continue

            # 重新构建映射（可能有些蛋白质被过滤掉）
            all_proteins = set(filtered_df['protein1']) | set(filtered_df['protein2'])
            all_proteins = sorted(list(all_proteins))

            protein_to_idx_sub = {protein: idx for idx, protein in enumerate(all_proteins)}

            # 构建边
            edge_index_list = []
            edge_attr_list = []

            for _, row in filtered_df.iterrows():
                idx1 = protein_to_idx_sub[row['protein1']]
                idx2 = protein_to_idx_sub[row['protein2']]
                score = row['combined_score'] / 1000.0

                edge_index_list.extend([[idx1, idx2], [idx2, idx1]])
                edge_attr_list.extend([score, score])

            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

            # 创建子图
            subgraph = Data(
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=len(protein_to_idx_sub),
                x=None
            )

            subgraph.protein_to_idx = protein_to_idx_sub
            subgraph.species = self.species_name
            subgraph.confidence_threshold = threshold

            subgraphs[threshold] = subgraph
            logger.info(f"Subgraph for threshold {threshold}: {len(all_proteins)} nodes, {edge_index.size(1)} edges")

        return subgraphs


def load_string_data(config_path: str) -> Data:
    """
    便捷函数：从配置文件加载STRING数据

    Args:
        config_path: 配置文件路径

    Returns:
        Data: 图数据对象
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    loader = STRINGLoader(config)
    return loader.load_and_process()


if __name__ == "__main__":
    # 示例用法
    import yaml
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "cfg/data_string_human.yaml"

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 加载数据
    graph_data = load_string_data(config_path)

    print(f"\nSTRING data loaded successfully!")
    print(f"Graph: {graph_data}")
