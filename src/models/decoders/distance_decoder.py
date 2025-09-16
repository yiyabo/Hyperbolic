"""
双曲距离解码器实现

该模块实现了基于双曲距离的链路预测解码器，包括：
- 双曲距离计算
- 温度缩放
- 多种评分函数
- 批量处理
- 数值稳定性保证

核心思想：
s_ij = -d_c(h_i, h_j) / τ
其中 d_c 是双曲距离，τ 是可学习温度参数
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class HyperbolicDistanceDecoder(nn.Module):
    """
    双曲距离解码器

    使用双曲距离计算节点对之间的相似性分数：
    - 距离越小，相似性分数越高
    - 支持温度缩放
    - 支持多种距离变换函数
    """

    def __init__(
        self,
        manifold,
        temperature: Union[float, Literal["learnable", "adaptive"]] = "learnable",
        distance_transform: Literal["negative", "exp_negative", "inv_distance"] = "negative",
        temperature_init: float = 1.0,
        temperature_min: float = 0.01,
        temperature_max: float = 10.0,
        use_bias: bool = False,
        eps: float = 1e-15
    ):
        super().__init__()

        self.manifold = manifold
        self.distance_transform = distance_transform
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.eps = eps

        # 温度参数设置
        if temperature == "learnable":
            # 可学习温度，使用 softplus 确保正值
            self.temperature_param = nn.Parameter(
                torch.tensor(math.log(math.exp(temperature_init) - 1))
            )
            self.temperature_mode = "learnable"
        elif temperature == "adaptive":
            # 自适应温度（根据距离分布动态调整）
            self.register_buffer('temperature_param', torch.tensor(temperature_init))
            self.temperature_mode = "adaptive"
            # 用于跟踪距离统计
            self.register_buffer('distance_ema', torch.tensor(1.0))
            self.register_buffer('update_count', torch.tensor(0))
            self.ema_momentum = 0.99
        else:
            # 固定温度
            self.register_buffer('temperature_param', torch.tensor(float(temperature)))
            self.temperature_mode = "fixed"

        # 可选的偏置项
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('bias', None)

        # 统计信息
        self.register_buffer('forward_count', torch.tensor(0))
        self.register_buffer('distance_sum', torch.tensor(0.0))
        self.register_buffer('min_distance', torch.tensor(float('inf')))
        self.register_buffer('max_distance', torch.tensor(0.0))

    @property
    def temperature(self) -> torch.Tensor:
        """获取当前温度参数"""
        if self.temperature_mode == "learnable":
            temp = F.softplus(self.temperature_param) + self.eps
        else:
            temp = self.temperature_param

        # 限制温度范围
        return torch.clamp(temp, min=self.temperature_min, max=self.temperature_max)

    def forward(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        return_distances: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播：计算节点对的相似性分数

        Args:
            h_i: (..., d+1) 源节点嵌入
            h_j: (..., d+1) 目标节点嵌入
            return_distances: 是否同时返回距离值

        Returns:
            scores: (...,) 相似性分数
            distances: (...,) 双曲距离 (如果 return_distances=True)
        """
        # 检查输入维度
        if h_i.shape != h_j.shape:
            raise ValueError(f"Input shapes must match: {h_i.shape} vs {h_j.shape}")

        if h_i.size(-1) < 2:
            raise ValueError(f"Input dimension must be at least 2, got {h_i.size(-1)}")

        # 更新统计
        self.forward_count += 1

        # 计算双曲距离
        distances = self.manifold.dist(h_i, h_j)  # (...,)

        # 更新距离统计
        self._update_distance_statistics(distances)

        # 自适应温度更新
        if self.temperature_mode == "adaptive" and self.training:
            self._update_adaptive_temperature(distances)

        # 应用距离变换和温度缩放
        scores = self._transform_distances_to_scores(distances)

        # 添加偏置
        if self.bias is not None:
            scores = scores + self.bias

        if return_distances:
            return scores, distances
        else:
            return scores

    def forward_pairs(
        self,
        h: torch.Tensor,
        pairs: torch.Tensor,
        return_distances: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        对指定节点对计算分数

        Args:
            h: (N, d+1) 所有节点嵌入
            pairs: (P, 2) 节点对索引
            return_distances: 是否返回距离

        Returns:
            scores: (P,) 分数
            distances: (P,) 距离 (如果需要)
        """
        h_i = h[pairs[:, 0]]  # (P, d+1)
        h_j = h[pairs[:, 1]]  # (P, d+1)

        return self.forward(h_i, h_j, return_distances=return_distances)

    def forward_edges(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        return_distances: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        对边列表计算分数

        Args:
            h: (N, d+1) 节点嵌入
            edge_index: (2, E) 边索引
            return_distances: 是否返回距离

        Returns:
            scores: (E,) 分数
            distances: (E,) 距离 (如果需要)
        """
        row, col = edge_index[0], edge_index[1]
        h_i = h[row]  # (E, d+1)
        h_j = h[col]  # (E, d+1)

        return self.forward(h_i, h_j, return_distances=return_distances)

    def score_all_pairs(
        self,
        h: torch.Tensor,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        计算所有节点对的分数（用于全图评估）

        Args:
            h: (N, d+1) 节点嵌入
            chunk_size: 分块大小，避免内存溢出

        Returns:
            scores: (N, N) 分数矩阵
        """
        N = h.size(0)

        if chunk_size is None or chunk_size >= N:
            # 一次性计算
            h_i = h.unsqueeze(1)  # (N, 1, d+1)
            h_j = h.unsqueeze(0)  # (1, N, d+1)
            scores = self.forward(h_i, h_j)  # (N, N)
            return scores
        else:
            # 分块计算
            scores = torch.zeros(N, N, device=h.device, dtype=h.dtype)

            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                h_i_chunk = h[i:end_i].unsqueeze(1)  # (chunk, 1, d+1)
                h_j_all = h.unsqueeze(0)  # (1, N, d+1)

                chunk_scores = self.forward(h_i_chunk, h_j_all)  # (chunk, N)
                scores[i:end_i] = chunk_scores

            return scores

    def _transform_distances_to_scores(self, distances: torch.Tensor) -> torch.Tensor:
        """将距离转换为分数"""
        temp = self.temperature

        if self.distance_transform == "negative":
            # s = -d / τ
            scores = -distances / temp

        elif self.distance_transform == "exp_negative":
            # s = exp(-d / τ)
            scores = torch.exp(-distances / temp)

        elif self.distance_transform == "inv_distance":
            # s = 1 / (1 + d * τ)
            scores = 1.0 / (1.0 + distances * temp)

        else:
            raise ValueError(f"Unknown distance transform: {self.distance_transform}")

        return scores

    def _update_distance_statistics(self, distances: torch.Tensor):
        """更新距离统计信息"""
        with torch.no_grad():
            batch_min = distances.min()
            batch_max = distances.max()
            batch_sum = distances.sum()

            self.distance_sum += batch_sum
            self.min_distance = torch.min(self.min_distance, batch_min)
            self.max_distance = torch.max(self.max_distance, batch_max)

    def _update_adaptive_temperature(self, distances: torch.Tensor):
        """更新自适应温度"""
        with torch.no_grad():
            # 使用距离的平均值来调整温度
            current_mean = distances.mean()

            if self.update_count == 0:
                self.distance_ema.copy_(current_mean)
            else:
                self.distance_ema = (self.ema_momentum * self.distance_ema +
                                   (1 - self.ema_momentum) * current_mean)

            self.update_count += 1

            # 根据距离分布调整温度：距离大时增加温度，距离小时减少温度
            target_temp = torch.clamp(self.distance_ema,
                                    min=self.temperature_min,
                                    max=self.temperature_max)

            # 平滑更新
            self.temperature_param = (0.9 * self.temperature_param + 0.1 * target_temp)

    def get_statistics(self) -> dict:
        """获取解码器统计信息"""
        stats = {
            'forward_count': int(self.forward_count),
            'current_temperature': float(self.temperature),
            'temperature_mode': self.temperature_mode,
            'distance_transform': self.distance_transform,
        }

        if self.forward_count > 0:
            avg_distance = float(self.distance_sum / self.forward_count)
            stats.update({
                'avg_distance': avg_distance,
                'min_distance': float(self.min_distance),
                'max_distance': float(self.max_distance),
            })

        return stats

    def reset_statistics(self):
        """重置统计信息"""
        self.forward_count.zero_()
        self.distance_sum.zero_()
        self.min_distance.fill_(float('inf'))
        self.max_distance.zero_()

        if self.temperature_mode == "adaptive":
            self.distance_ema.zero_()
            self.update_count.zero_()

    def extra_repr(self) -> str:
        return (f'temperature_mode={self.temperature_mode}, '
                f'distance_transform={self.distance_transform}, '
                f'current_temp={self.temperature:.4f}')


class BilinearDecoder(nn.Module):
    """
    双线性解码器（用于消融实验对比）

    在切空间中执行双线性操作：
    s_ij = log_o(h_i)^T R log_o(h_j)
    """

    def __init__(
        self,
        manifold,
        feature_dim: int,
        use_bias: bool = True,
        basepoint: Literal["origin", "adaptive"] = "origin"
    ):
        super().__init__()

        self.manifold = manifold
        self.feature_dim = feature_dim  # 切空间维度
        self.basepoint_mode = basepoint

        # 关系矩阵
        self.relation_matrix = nn.Parameter(torch.empty(feature_dim, feature_dim))

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.relation_matrix)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _get_basepoint(self, h: torch.Tensor) -> torch.Tensor:
        """获取基点"""
        if self.basepoint_mode == "origin":
            c = self.manifold.c
            basepoint = torch.zeros_like(h)
            basepoint[..., 0] = 1.0 / torch.sqrt(c)
            return basepoint
        else:
            # adaptive: 使用输入的均值作为基点
            return h.mean(dim=-2, keepdim=True).expand_as(h)

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            h_i: (..., d+1) 源节点嵌入
            h_j: (..., d+1) 目标节点嵌入

        Returns:
            scores: (...,) 相似性分数
        """
        # 获取基点
        basepoint = self._get_basepoint(h_i)

        # 映射到切空间
        h_i_tan = self.manifold.log(basepoint, h_i)  # (..., d+1)
        h_j_tan = self.manifold.log(basepoint, h_j)  # (..., d+1)

        # 转换为纯切空间表示
        h_i_pure = self.manifold.from_tangent_dplus1(h_i_tan)  # (..., d)
        h_j_pure = self.manifold.from_tangent_dplus1(h_j_tan)  # (..., d)

        # 双线性操作: h_i^T R h_j
        h_i_transformed = torch.matmul(h_i_pure, self.relation_matrix)  # (..., d)
        scores = torch.sum(h_i_transformed * h_j_pure, dim=-1)  # (...,)

        # 添加偏置
        if self.bias is not None:
            scores = scores + self.bias

        return scores

    def forward_pairs(self, h: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """对指定节点对计算分数"""
        h_i = h[pairs[:, 0]]  # (P, d+1)
        h_j = h[pairs[:, 1]]  # (P, d+1)
        return self.forward(h_i, h_j)

    def forward_edges(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """对边列表计算分数"""
        row, col = edge_index[0], edge_index[1]
        h_i = h[row]  # (E, d+1)
        h_j = h[col]  # (E, d+1)
        return self.forward(h_i, h_j)


class DotProductDecoder(nn.Module):
    """
    点积解码器（简单基线）

    在切空间中执行点积：
    s_ij = log_o(h_i)^T log_o(h_j)
    """

    def __init__(self, manifold, use_bias: bool = True):
        super().__init__()

        self.manifold = manifold

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('bias', None)

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 基点
        c = self.manifold.c
        basepoint = torch.zeros_like(h_i)
        basepoint[..., 0] = 1.0 / torch.sqrt(c)

        # 映射到切空间
        h_i_tan = self.manifold.log(basepoint, h_i)
        h_j_tan = self.manifold.log(basepoint, h_j)

        # 切空间点积
        scores = torch.sum(h_i_tan * h_j_tan, dim=-1)

        if self.bias is not None:
            scores = scores + self.bias

        return scores

    def forward_pairs(self, h: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """对指定节点对计算分数"""
        h_i = h[pairs[:, 0]]
        h_j = h[pairs[:, 1]]
        return self.forward(h_i, h_j)

    def forward_edges(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """对边列表计算分数"""
        row, col = edge_index[0], edge_index[1]
        h_i = h[row]
        h_j = h[col]
        return self.forward(h_i, h_j)


def create_decoder(
    manifold,
    decoder_type: str = "distance",
    **kwargs
) -> nn.Module:
    """
    创建解码器的便捷函数

    Args:
        manifold: 流形对象
        decoder_type: 解码器类型
        **kwargs: 解码器特定参数

    Returns:
        解码器实例
    """
    decoder_map = {
        "distance": HyperbolicDistanceDecoder,
        "bilinear": BilinearDecoder,
        "dot_product": DotProductDecoder,
    }

    if decoder_type not in decoder_map:
        raise ValueError(f"Unknown decoder type: {decoder_type}. "
                        f"Supported types: {list(decoder_map.keys())}")

    return decoder_map[decoder_type](manifold, **kwargs)
