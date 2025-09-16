"""
Lorentz 聚合器实现

该模块实现了双曲空间中的邻域聚合操作，包括：
- Lorentz 加权聚合
- 时样性检查和兜底机制
- 多种聚合模式支持
- 数值稳定性保证
- 上片约束维护

核心思想：
- 使用 Minkowski 加权和后归一化
- 若结果非时样则回退到切空间均值
- 确保聚合结果满足双曲约束
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax
from typing import Optional, Union, Literal
import logging

logger = logging.getLogger(__name__)


class LorentzAggregator(nn.Module):
    """
    Lorentz 流形上的聚合器

    支持多种聚合模式：
    - mean: 度归一化的加权平均
    - attention: 基于注意力的加权聚合
    - max: 最大池化（在切空间执行）
    - sum: 简单求和后归一化
    """

    def __init__(
        self,
        manifold,
        mode: Literal["mean", "attention", "max", "sum"] = "mean",
        basepoint: Literal["origin", "nodewise"] = "origin",
        attention_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()

        self.manifold = manifold
        self.mode = mode
        self.basepoint = basepoint
        self.dropout = dropout

        # 注意力机制参数
        if mode == "attention":
            if attention_dim is None:
                raise ValueError("attention_dim must be specified for attention mode")
            self.attention_dim = attention_dim
            # 注意力层在切空间操作
            self.attention_linear = nn.Linear(attention_dim - 1, 1, bias=False)  # d维度（去掉x0）
            self.attention_dropout = nn.Dropout(dropout)

        # 统计信息
        self.register_buffer('fallback_count', torch.tensor(0))
        self.register_buffer('total_count', torch.tensor(0))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        size: Optional[int] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (N, d+1) 节点特征在双曲空间
            edge_index: (2, E) 边索引 [source, target]
            alpha: (E,) 可选的边权重，会被归一化
            size: 目标节点数量，默认为x.size(0)

        Returns:
            (N, d+1) 聚合结果，满足双曲约束
        """
        if size is None:
            size = x.size(0)

        row, col = edge_index[0], edge_index[1]

        # 获取邻居特征
        x_neighbors = x[row]  # (E, d+1)

        # 计算聚合权重
        if self.mode == "mean":
            weights = self._compute_mean_weights(edge_index, size, alpha)
        elif self.mode == "attention":
            weights = self._compute_attention_weights(x, x_neighbors, col, alpha)
        elif self.mode == "max":
            return self._compute_max_aggregation(x, x_neighbors, col, size)
        elif self.mode == "sum":
            weights = self._compute_sum_weights(edge_index, size, alpha)
        else:
            raise ValueError(f"Unknown aggregation mode: {self.mode}")

        # 执行 Lorentz 加权聚合
        aggregated = self._lorentz_weighted_aggregation(
            x_neighbors, weights, col, size
        )

        return aggregated

    def _compute_mean_weights(
        self,
        edge_index: torch.Tensor,
        size: int,
        alpha: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算度归一化权重"""
        row, col = edge_index[0], edge_index[1]

        if alpha is not None:
            # 使用提供的权重
            weights = alpha
        else:
            # 均匀权重
            weights = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float)

        # 按目标节点归一化（row-wise softmax等价）
        weights = softmax(torch.zeros_like(weights), col, num_nodes=size) * weights

        # 度归一化
        deg = scatter(weights, col, dim=0, dim_size=size, reduce='sum')
        deg = torch.clamp(deg, min=1e-12)
        weights = weights / deg[col]

        return weights

    def _compute_attention_weights(
        self,
        x_target: torch.Tensor,
        x_neighbors: torch.Tensor,
        col: torch.Tensor,
        alpha: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算注意力权重"""
        # 转换到切空间进行注意力计算
        basepoint = self._get_basepoint(x_target, col)

        # 目标节点到切空间
        x_target_tan = self.manifold.log(basepoint, x_target[col])  # (E, d+1)
        x_target_tan = self.manifold.from_tangent_dplus1(x_target_tan)  # (E, d)

        # 邻居节点到切空间
        x_neighbors_tan = self.manifold.log(basepoint, x_neighbors)  # (E, d+1)
        x_neighbors_tan = self.manifold.from_tangent_dplus1(x_neighbors_tan)  # (E, d)

        # 计算注意力分数（简单的点积注意力）
        attention_input = x_target_tan + x_neighbors_tan  # 可以用concat或其他方式
        attention_scores = self.attention_linear(attention_input).squeeze(-1)  # (E,)

        # Dropout
        attention_scores = self.attention_dropout(attention_scores)

        # 结合预设权重
        if alpha is not None:
            attention_scores = attention_scores + torch.log(alpha + 1e-16)

        # Softmax 归一化
        weights = softmax(attention_scores, col, num_nodes=x_target.size(0))

        return weights

    def _compute_max_aggregation(
        self,
        x_target: torch.Tensor,
        x_neighbors: torch.Tensor,
        col: torch.Tensor,
        size: int
    ) -> torch.Tensor:
        """最大池化聚合（在切空间执行）"""
        basepoint = self._get_basepoint(x_target, col)

        # 转到切空间
        x_neighbors_tan = self.manifold.log(basepoint, x_neighbors)
        x_neighbors_tan = self.manifold.from_tangent_dplus1(x_neighbors_tan)  # (E, d)

        # 切空间最大池化
        max_tan = scatter(x_neighbors_tan, col, dim=0, dim_size=size, reduce='max')  # (N, d)

        # 转回双曲空间
        max_tan_dplus1 = self.manifold.to_tangent_dplus1(max_tan)
        result = self.manifold.exp(basepoint[:size], max_tan_dplus1)

        return result

    def _compute_sum_weights(
        self,
        edge_index: torch.Tensor,
        size: int,
        alpha: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算求和权重（不归一化，后续会在聚合中处理）"""
        if alpha is not None:
            return alpha
        else:
            return torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float)

    def _get_basepoint(self, x_target: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        """获取基点"""
        if self.basepoint == "origin":
            # 统一使用原点作为基点
            c = self.manifold.c
            basepoint = torch.zeros_like(x_target)
            basepoint[..., 0] = 1.0 / torch.sqrt(c)
            return basepoint
        elif self.basepoint == "nodewise":
            # 使用目标节点作为基点
            return x_target[col]
        else:
            raise ValueError(f"Unknown basepoint mode: {self.basepoint}")

    def _lorentz_weighted_aggregation(
        self,
        x_neighbors: torch.Tensor,
        weights: torch.Tensor,
        col: torch.Tensor,
        size: int
    ) -> torch.Tensor:
        """
        执行 Lorentz 加权聚合

        核心算法：
        1. Minkowski 加权和
        2. 时样性检查
        3. 若非时样则切空间兜底
        4. 归一化到双曲面
        5. 上片修正
        """
        # 更新统计
        self.total_count += 1

        # 加权求和
        weighted_neighbors = weights.unsqueeze(-1) * x_neighbors  # (E, d+1)
        aggregated = scatter(weighted_neighbors, col, dim=0, dim_size=size, reduce='sum')  # (N, d+1)

        # 检查时样性 - 放宽阈值，避免过度触发兜底机制
        lorentz_norm_sq = -self.manifold.dot(aggregated, aggregated)  # (N,)

        # 找出非时样的节点 - 使用更宽松的阈值
        non_timelike_mask = lorentz_norm_sq <= 1e-8

        if non_timelike_mask.any():
            logger.debug(f"Found {non_timelike_mask.sum()} non-timelike aggregations, using fallback")
            self.fallback_count += non_timelike_mask.sum()

            # 对非时样节点使用高效的兜底机制
            aggregated = self._fallback_tangent_aggregation_optimized(
                aggregated, x_neighbors, weights, col, size, non_timelike_mask
            )

            # 重新计算范数
            lorentz_norm_sq = -self.manifold.dot(aggregated, aggregated)

        # 归一化到双曲面 ||x||_L = 1/√c
        c = self.manifold.c
        denom = torch.clamp(torch.sqrt(lorentz_norm_sq), min=1e-12)  # 更安全的最小值
        normalized = aggregated / (denom.unsqueeze(-1) * torch.sqrt(c))

        # 上片修正 (x₀ > 0)
        negative_x0_mask = normalized[..., 0] <= 0
        if negative_x0_mask.any():
            normalized[negative_x0_mask] = -normalized[negative_x0_mask]

        return normalized

    def _fallback_tangent_aggregation_optimized(
        self,
        failed_aggregated: torch.Tensor,
        x_neighbors: torch.Tensor,
        weights: torch.Tensor,
        col: torch.Tensor,
        size: int,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """优化的切空间兜底聚合 - 批量处理提高效率"""
        result = failed_aggregated.clone()

        # 获取基点（优化：批量创建）
        c = self.manifold.c
        if self.basepoint == "origin":
            basepoint = torch.zeros(size, failed_aggregated.size(-1),
                                  device=failed_aggregated.device, dtype=failed_aggregated.dtype)
            basepoint[..., 0] = 1.0 / torch.sqrt(c)
        else:
            # 简化的nodewise基点处理
            basepoint = torch.zeros_like(result)
            unique_targets = torch.unique(col[mask[col]])
            for target in unique_targets:
                neighbor_indices = (col == target).nonzero(as_tuple=True)[0]
                if len(neighbor_indices) > 0:
                    basepoint[target] = x_neighbors[neighbor_indices[0]]

        # 批量处理失败的节点
        failed_indices = mask.nonzero(as_tuple=True)[0]

        if len(failed_indices) > 0:
            # 为每个失败节点收集邻居信息
            for target_idx in failed_indices:
                neighbor_mask = col == target_idx
                if not neighbor_mask.any():
                    # 如果没有邻居，使用基点本身
                    result[target_idx] = basepoint[target_idx]
                    continue

                target_neighbors = x_neighbors[neighbor_mask]  # (degree, d+1)
                target_weights = weights[neighbor_mask]  # (degree,)

                # 归一化权重
                weight_sum = target_weights.sum()
                if weight_sum > 1e-12:
                    target_weights = target_weights / weight_sum
                else:
                    target_weights = torch.ones_like(target_weights) / len(target_weights)

                # 基点
                base = basepoint[target_idx].unsqueeze(0)  # (1, d+1)

                # 批量对数映射
                try:
                    tangent_vectors = self.manifold.log(base, target_neighbors)  # (degree, d+1)

                    # 加权平均
                    weighted_tangent = torch.sum(target_weights.unsqueeze(-1) * tangent_vectors, dim=0)

                    # 指数映射回双曲空间
                    result[target_idx] = self.manifold.exp(base, weighted_tangent.unsqueeze(0)).squeeze(0)
                except Exception as e:
                    # 如果出现数值问题，使用更安全的备用方案
                    logger.debug(f"Numerical issue in fallback aggregation for node {target_idx}: {e}")
                    result[target_idx] = basepoint[target_idx]

        return result

    def _fallback_tangent_aggregation(
        self,
        failed_aggregated: torch.Tensor,
        x_neighbors: torch.Tensor,
        weights: torch.Tensor,
        col: torch.Tensor,
        size: int,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """原始切空间兜底聚合（保留向后兼容）"""
        return self._fallback_tangent_aggregation_optimized(
            failed_aggregated, x_neighbors, weights, col, size, mask
        )

    def get_fallback_ratio(self) -> float:
        """获取兜底机制使用率"""
        if self.total_count == 0:
            return 0.0
        return float(self.fallback_count) / float(self.total_count)

    def reset_statistics(self):
        """重置统计信息"""
        self.fallback_count.zero_()
        self.total_count.zero_()


class HyperbolicAttention(nn.Module):
    """双曲注意力机制"""

    def __init__(self, manifold, dim: int, heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.manifold = manifold
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert dim % heads == 0

        # 在切空间进行注意力计算
        self.query_proj = nn.Linear(dim - 1, dim - 1, bias=False)  # 去掉x0维度
        self.key_proj = nn.Linear(dim - 1, dim - 1, bias=False)
        self.value_proj = nn.Linear(dim - 1, dim - 1, bias=False)
        self.out_proj = nn.Linear(dim - 1, dim - 1)

        self.dropout = nn.Dropout(dropout)
        self.scale = (self.head_dim) ** -0.5

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, d+1) 节点特征
            edge_index: (2, E) 边索引

        Returns:
            (N, d+1) 注意力聚合结果
        """
        N = x.size(0)
        row, col = edge_index[0], edge_index[1]

        # 转换到切空间（使用原点作为基点）
        c = self.manifold.c
        origin = torch.zeros_like(x)
        origin[..., 0] = 1.0 / torch.sqrt(c)

        x_tan = self.manifold.log(origin, x)  # (N, d+1)
        x_tan = self.manifold.from_tangent_dplus1(x_tan)  # (N, d)

        # 多头注意力投影
        Q = self.query_proj(x_tan).view(N, self.heads, self.head_dim)  # (N, H, D)
        K = self.key_proj(x_tan).view(N, self.heads, self.head_dim)  # (N, H, D)
        V = self.value_proj(x_tan).view(N, self.heads, self.head_dim)  # (N, H, D)

        # 计算边上的注意力分数
        query_edges = Q[col]  # (E, H, D)
        key_edges = K[row]    # (E, H, D)
        value_edges = V[row]  # (E, H, D)

        # 注意力分数
        att_scores = torch.sum(query_edges * key_edges, dim=-1) * self.scale  # (E, H)
        att_scores = softmax(att_scores.view(-1), col.repeat_interleave(self.heads), num_nodes=N)
        att_scores = att_scores.view(-1, self.heads, 1)  # (E, H, 1)
        att_scores = self.dropout(att_scores)

        # 加权聚合
        weighted_values = att_scores * value_edges  # (E, H, D)
        aggregated = scatter(weighted_values, col.unsqueeze(-1).unsqueeze(-1).expand_as(weighted_values),
                           dim=0, dim_size=N, reduce='sum')  # (N, H, D)

        # 合并多头
        aggregated = aggregated.view(N, -1)  # (N, H*D)
        aggregated = self.out_proj(aggregated)  # (N, d)

        # 转换回双曲空间
        aggregated_dplus1 = self.manifold.to_tangent_dplus1(aggregated)  # (N, d+1)
        result = self.manifold.exp(origin, aggregated_dplus1)

        return result


def create_aggregator(
    manifold,
    mode: str = "mean",
    **kwargs
) -> LorentzAggregator:
    """创建聚合器的便捷函数"""
    return LorentzAggregator(manifold=manifold, mode=mode, **kwargs)
