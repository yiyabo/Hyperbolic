"""
负采样模块

该模块实现多种负采样策略，包括：
- 均匀负采样（随机采样）
- 拓扑驱动的难负采样
- 基于度数相似性的负采样
- 基于共同邻居的负采样
- 基于社团结构的负采样
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, negative_sampling
from typing import Dict, List, Tuple, Optional, Set
import logging
import random
from collections import defaultdict
import community as community_detection

logger = logging.getLogger(__name__)


class NegativeSampler:
    """负采样器基类"""

    def __init__(self, strategy: str = "uniform", ratio: int = 5,
                 hard_frac: float = 0.5, seed: int = 42):
        """
        初始化负采样器

        Args:
            strategy: 采样策略 ("uniform", "topology_driven", "degree_similar", "common_neighbor")
            ratio: 负正样本比例
            hard_frac: 难负样本比例（仅对topology_driven有效）
            seed: 随机种子
        """
        self.strategy = strategy
        self.ratio = ratio
        self.hard_frac = hard_frac
        self.seed = seed

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 缓存图相关信息
        self.graph = None
        self.nx_graph = None
        self.node_degrees = None
        self.communities = None
        self.common_neighbors_cache = {}

        logger.info(f"Initialized NegativeSampler with strategy='{strategy}', ratio={ratio}, hard_frac={hard_frac}")

    def prepare_graph(self, graph_data: Data):
        """
        预处理图数据，计算必要的拓扑信息

        Args:
            graph_data: PyTorch Geometric图数据
        """
        logger.info("Preparing graph for negative sampling...")

        self.graph = graph_data

        # 计算节点度数
        edge_index = graph_data.edge_index
        self.node_degrees = torch.bincount(edge_index[0], minlength=graph_data.num_nodes)

        if self.strategy in ["topology_driven", "degree_similar", "common_neighbor", "community_based"]:
            # 转换为NetworkX图（用于复杂拓扑分析）
            self.nx_graph = to_networkx(graph_data, to_undirected=True)

            # 计算社团结构
            if self.strategy in ["topology_driven", "community_based"]:
                self._compute_communities()

        logger.info(f"Graph prepared: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")

    def _compute_communities(self):
        """计算图的社团结构"""
        try:
            logger.info("Computing community structure...")
            partition = community_detection.best_partition(self.nx_graph)

            # 将社团信息组织为字典
            self.communities = defaultdict(list)
            for node, community_id in partition.items():
                self.communities[community_id].append(node)

            logger.info(f"Found {len(self.communities)} communities")

        except Exception as e:
            logger.warning(f"Community detection failed: {e}, using degree-based grouping")
            self._fallback_community_detection()

    def _fallback_community_detection(self):
        """备用社团检测：基于度数分组"""
        degree_groups = defaultdict(list)

        for node in range(self.graph.num_nodes):
            degree = int(self.node_degrees[node])
            # 按度数分组（每个组包含相似度数的节点）
            group_id = degree // 10  # 每10个度数一组
            degree_groups[group_id].append(node)

        self.communities = degree_groups
        logger.info(f"Created {len(self.communities)} degree-based groups")

    def sample(self, pos_edges: torch.Tensor, num_neg_samples: Optional[int] = None) -> torch.Tensor:
        """
        采样负边

        Args:
            pos_edges: 正边，shape为[2, num_pos_edges]
            num_neg_samples: 负样本数量，如果为None则使用ratio计算

        Returns:
            torch.Tensor: 负边，shape为[2, num_neg_samples]
        """
        if num_neg_samples is None:
            num_neg_samples = pos_edges.size(1) * self.ratio

        logger.debug(f"Sampling {num_neg_samples} negative edges using strategy '{self.strategy}'")

        if self.strategy == "uniform":
            return self._sample_uniform(num_neg_samples)
        elif self.strategy == "topology_driven":
            return self._sample_topology_driven(pos_edges, num_neg_samples)
        elif self.strategy == "degree_similar":
            return self._sample_degree_similar(pos_edges, num_neg_samples)
        elif self.strategy == "common_neighbor":
            return self._sample_common_neighbor(pos_edges, num_neg_samples)
        elif self.strategy == "community_based":
            return self._sample_community_based(pos_edges, num_neg_samples)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")

    def _sample_uniform(self, num_neg_samples: int) -> torch.Tensor:
        """均匀负采样"""
        return negative_sampling(
            edge_index=self.graph.edge_index,
            num_nodes=self.graph.num_nodes,
            num_neg_samples=num_neg_samples
        )

    def _sample_topology_driven(self, pos_edges: torch.Tensor, num_neg_samples: int) -> torch.Tensor:
        """拓扑驱动的混合负采样"""
        num_hard = int(num_neg_samples * self.hard_frac)
        num_easy = num_neg_samples - num_hard

        # 采样难负样本（混合多种策略）
        hard_neg_edges = []

        # 1/3 基于度数相似性
        if num_hard > 0:
            num_degree = max(1, num_hard // 3)
            degree_negs = self._sample_degree_similar(pos_edges, num_degree)
            hard_neg_edges.append(degree_negs)

        # 1/3 基于共同邻居
        if num_hard > 0:
            num_common = max(1, num_hard // 3)
            common_negs = self._sample_common_neighbor(pos_edges, num_common)
            hard_neg_edges.append(common_negs)

        # 1/3 基于社团结构
        if num_hard > 0:
            num_community = num_hard - (len(hard_neg_edges) * max(1, num_hard // 3))
            if num_community > 0:
                community_negs = self._sample_community_based(pos_edges, num_community)
                hard_neg_edges.append(community_negs)

        # 合并难负样本
        if hard_neg_edges:
            hard_negs = torch.cat(hard_neg_edges, dim=1)
            # 如果数量不足，用度数相似性补充
            if hard_negs.size(1) < num_hard:
                extra_needed = num_hard - hard_negs.size(1)
                extra_negs = self._sample_degree_similar(pos_edges, extra_needed)
                hard_negs = torch.cat([hard_negs, extra_negs], dim=1)
            elif hard_negs.size(1) > num_hard:
                # 随机选择指定数量
                indices = torch.randperm(hard_negs.size(1))[:num_hard]
                hard_negs = hard_negs[:, indices]
        else:
            hard_negs = torch.empty((2, 0), dtype=torch.long)

        # 采样简单负样本（均匀采样）
        easy_negs = self._sample_uniform(num_easy)

        # 合并所有负样本
        if hard_negs.size(1) > 0 and easy_negs.size(1) > 0:
            all_negs = torch.cat([hard_negs, easy_negs], dim=1)
        elif hard_negs.size(1) > 0:
            all_negs = hard_negs
        else:
            all_negs = easy_negs

        # 随机打乱
        if all_negs.size(1) > 0:
            perm = torch.randperm(all_negs.size(1))
            all_negs = all_negs[:, perm]

        return all_negs

    def _sample_degree_similar(self, pos_edges: torch.Tensor, num_neg_samples: int) -> torch.Tensor:
        """基于度数相似性的负采样"""
        neg_edges = []
        existing_edges = set()

        # 构建现有边的集合
        for i in range(self.graph.edge_index.size(1)):
            u, v = int(self.graph.edge_index[0, i]), int(self.graph.edge_index[1, i])
            existing_edges.add((min(u, v), max(u, v)))

        attempts = 0
        max_attempts = num_neg_samples * 10

        while len(neg_edges) < num_neg_samples and attempts < max_attempts:
            # 随机选择一个正边
            pos_idx = random.randint(0, pos_edges.size(1) - 1)
            u, v = int(pos_edges[0, pos_idx]), int(pos_edges[1, pos_idx])

            # 获取u和v的度数
            degree_u = int(self.node_degrees[u])
            degree_v = int(self.node_degrees[v])

            # 寻找度数相似的节点对
            candidates_u = self._find_degree_similar_nodes(u, degree_u)
            candidates_v = self._find_degree_similar_nodes(v, degree_v)

            if candidates_u and candidates_v:
                new_u = random.choice(candidates_u)
                new_v = random.choice(candidates_v)

                if new_u != new_v:
                    edge = (min(new_u, new_v), max(new_u, new_v))
                    if edge not in existing_edges:
                        neg_edges.append([new_u, new_v])
                        existing_edges.add(edge)

            attempts += 1

        # 如果难负样本不足，用随机采样补充
        if len(neg_edges) < num_neg_samples:
            remaining = num_neg_samples - len(neg_edges)
            random_negs = self._sample_uniform(remaining)

            for i in range(random_negs.size(1)):
                neg_edges.append([int(random_negs[0, i]), int(random_negs[1, i])])

        if not neg_edges:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(neg_edges[:num_neg_samples], dtype=torch.long).t()

    def _find_degree_similar_nodes(self, node: int, target_degree: int, tolerance: int = 2) -> List[int]:
        """查找度数相似的节点"""
        similar_nodes = []

        for candidate in range(self.graph.num_nodes):
            if candidate != node:
                candidate_degree = int(self.node_degrees[candidate])
                if abs(candidate_degree - target_degree) <= tolerance:
                    similar_nodes.append(candidate)

        return similar_nodes

    def _sample_common_neighbor(self, pos_edges: torch.Tensor, num_neg_samples: int) -> torch.Tensor:
        """基于共同邻居的负采样"""
        if self.nx_graph is None:
            return self._sample_uniform(num_neg_samples)

        neg_edges = []
        existing_edges = set()

        # 构建现有边的集合
        for u, v in self.nx_graph.edges():
            existing_edges.add((min(u, v), max(u, v)))

        attempts = 0
        max_attempts = num_neg_samples * 10

        while len(neg_edges) < num_neg_samples and attempts < max_attempts:
            # 随机选择两个节点
            u = random.randint(0, self.graph.num_nodes - 1)
            v = random.randint(0, self.graph.num_nodes - 1)

            if u != v and (min(u, v), max(u, v)) not in existing_edges:
                # 计算共同邻居数量
                neighbors_u = set(self.nx_graph.neighbors(u))
                neighbors_v = set(self.nx_graph.neighbors(v))
                common_neighbors = len(neighbors_u & neighbors_v)

                # 偏好有共同邻居但未连接的节点对（难负样本）
                if common_neighbors > 0:
                    neg_edges.append([u, v])
                    existing_edges.add((min(u, v), max(u, v)))

            attempts += 1

        # 如果难负样本不足，用随机采样补充
        if len(neg_edges) < num_neg_samples:
            remaining = num_neg_samples - len(neg_edges)
            random_negs = self._sample_uniform(remaining)

            for i in range(random_negs.size(1)):
                neg_edges.append([int(random_negs[0, i]), int(random_negs[1, i])])

        if not neg_edges:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(neg_edges[:num_neg_samples], dtype=torch.long).t()

    def _sample_community_based(self, pos_edges: torch.Tensor, num_neg_samples: int) -> torch.Tensor:
        """基于社团结构的负采样"""
        if self.communities is None:
            return self._sample_uniform(num_neg_samples)

        neg_edges = []
        existing_edges = set()

        # 构建现有边的集合
        for i in range(self.graph.edge_index.size(1)):
            u, v = int(self.graph.edge_index[0, i]), int(self.graph.edge_index[1, i])
            existing_edges.add((min(u, v), max(u, v)))

        attempts = 0
        max_attempts = num_neg_samples * 10

        # 获取所有社团
        community_list = list(self.communities.values())
        community_pairs = []

        # 创建社团对（偏好同一社团内的节点对）
        for community in community_list:
            if len(community) >= 2:
                community_pairs.extend([(c, c) for c in [community]])  # 同一社团

        while len(neg_edges) < num_neg_samples and attempts < max_attempts:
            if community_pairs:
                # 优先从同一社团内采样
                comm1, comm2 = random.choice(community_pairs)
                if len(comm1) > 1:
                    u = random.choice(comm1)
                    v = random.choice(comm2)

                    if u != v and (min(u, v), max(u, v)) not in existing_edges:
                        neg_edges.append([u, v])
                        existing_edges.add((min(u, v), max(u, v)))
            else:
                # 回退到随机采样
                break

            attempts += 1

        # 如果社团内采样不足，用随机采样补充
        if len(neg_edges) < num_neg_samples:
            remaining = num_neg_samples - len(neg_edges)
            random_negs = self._sample_uniform(remaining)

            for i in range(random_negs.size(1)):
                neg_edges.append([int(random_negs[0, i]), int(random_negs[1, i])])

        if not neg_edges:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(neg_edges[:num_neg_samples], dtype=torch.long).t()

    def get_sampling_statistics(self, pos_edges: torch.Tensor, neg_edges: torch.Tensor) -> Dict:
        """
        获取采样统计信息

        Args:
            pos_edges: 正边
            neg_edges: 负边

        Returns:
            Dict: 统计信息
        """
        stats = {
            'num_positive_edges': pos_edges.size(1),
            'num_negative_edges': neg_edges.size(1),
            'negative_positive_ratio': neg_edges.size(1) / pos_edges.size(1) if pos_edges.size(1) > 0 else 0,
            'sampling_strategy': self.strategy
        }

        if self.strategy == "topology_driven":
            stats['hard_fraction'] = self.hard_frac

        # 计算度数统计
        if neg_edges.size(1) > 0:
            neg_degrees_u = self.node_degrees[neg_edges[0]]
            neg_degrees_v = self.node_degrees[neg_edges[1]]

            stats['neg_avg_degree_u'] = float(neg_degrees_u.float().mean())
            stats['neg_avg_degree_v'] = float(neg_degrees_v.float().mean())

        if pos_edges.size(1) > 0:
            pos_degrees_u = self.node_degrees[pos_edges[0]]
            pos_degrees_v = self.node_degrees[pos_edges[1]]

            stats['pos_avg_degree_u'] = float(pos_degrees_u.float().mean())
            stats['pos_avg_degree_v'] = float(pos_degrees_v.float().mean())

        return stats


class AdaptiveNegativeSampler(NegativeSampler):
    """自适应负采样器，根据训练进度调整采样策略"""

    def __init__(self, initial_strategy: str = "uniform", final_strategy: str = "topology_driven",
                 transition_epochs: int = 50, **kwargs):
        """
        初始化自适应负采样器

        Args:
            initial_strategy: 初始采样策略
            final_strategy: 最终采样策略
            transition_epochs: 过渡轮数
            **kwargs: 其他参数传递给基类
        """
        super().__init__(strategy=initial_strategy, **kwargs)
        self.initial_strategy = initial_strategy
        self.final_strategy = final_strategy
        self.transition_epochs = transition_epochs
        self.current_epoch = 0

    def update_epoch(self, epoch: int):
        """更新当前训练轮数"""
        self.current_epoch = epoch

        # 计算当前应该使用的策略
        if epoch < self.transition_epochs:
            progress = epoch / self.transition_epochs

            # 简单的策略：前半段用初始策略，后半段逐渐过渡到最终策略
            if progress < 0.5:
                self.strategy = self.initial_strategy
            else:
                self.strategy = self.final_strategy
                # 逐渐增加难负样本比例
                self.hard_frac = min(0.5, (progress - 0.5) * 2 * 0.5)
        else:
            self.strategy = self.final_strategy

    def sample(self, pos_edges: torch.Tensor, num_neg_samples: Optional[int] = None) -> torch.Tensor:
        """重写采样方法，考虑当前策略"""
        logger.debug(f"Epoch {self.current_epoch}: using strategy '{self.strategy}' with hard_frac={self.hard_frac:.2f}")
        return super().sample(pos_edges, num_neg_samples)


def create_negative_sampler(config: Dict) -> NegativeSampler:
    """
    根据配置创建负采样器

    Args:
        config: 负采样配置

    Returns:
        NegativeSampler: 负采样器实例
    """
    strategy = config.get('strategy', 'uniform')
    ratio = config.get('ratio', 5)
    hard_frac = config.get('hard_frac', 0.5)
    seed = config.get('seed', 42)

    # 检查是否使用自适应采样
    if config.get('adaptive', False):
        return AdaptiveNegativeSampler(
            initial_strategy=config.get('initial_strategy', 'uniform'),
            final_strategy=config.get('final_strategy', 'topology_driven'),
            transition_epochs=config.get('transition_epochs', 50),
            ratio=ratio,
            hard_frac=hard_frac,
            seed=seed
        )
    else:
        return NegativeSampler(
            strategy=strategy,
            ratio=ratio,
            hard_frac=hard_frac,
            seed=seed
        )


if __name__ == "__main__":
    # 测试负采样器
    import logging
    from torch_geometric.utils import erdos_renyi_graph

    logging.basicConfig(level=logging.INFO)

    # 创建测试图
    num_nodes = 1000
    edge_index = erdos_renyi_graph(num_nodes, 0.01)
    graph_data = Data(edge_index=edge_index, num_nodes=num_nodes)

    # 测试不同策略
    strategies = ["uniform", "topology_driven", "degree_similar", "common_neighbor"]

    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")

        sampler = NegativeSampler(strategy=strategy, ratio=2, hard_frac=0.5)
        sampler.prepare_graph(graph_data)

        # 采样测试
        pos_edges = edge_index[:, :100]  # 取前100条边作为正样本
        neg_edges = sampler.sample(pos_edges)

        print(f"Positive edges: {pos_edges.size(1)}")
        print(f"Negative edges: {neg_edges.size(1)}")

        # 统计信息
        stats = sampler.get_sampling_statistics(pos_edges, neg_edges)
        for key, value in stats.items():
            print(f"  {key}: {value}")
