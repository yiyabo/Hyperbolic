"""
HGCN (双曲图卷积网络) 核心层实现

该模块实现了HGCN的核心卷积层，包括：
- 完整的HGCN前向传播流程
- 可配置的聚合和激活方式
- 数值稳定性保证
- 消融实验支持
- 残差连接

核心流程：
1. 切空间投影: log_o^(c)(X)
2. 线性变换: X W + b (在切空间)
3. 双曲投影: exp_o^(c)(H)
4. 邻域聚合: LorentzAgg(X_hyp, edge_index)
5. 激活函数: σ_H(X_tilde)
6. 超曲面归一化: 确保 <x,x>_L = -1/c
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import Optional, Union, Literal, Tuple
import logging

from .aggregators import LorentzAggregator, create_aggregator
from .activations import create_hyperbolic_activation, HyperbolicDropout
from .linear_layer import HyperbolicLinear

logger = logging.getLogger(__name__)


class HGCNLayer(MessagePassing):
    """
    HGCN (双曲图卷积网络) 层

    实现完整的双曲图卷积操作，包括特征变换、邻域聚合和非线性激活
    """

    def __init__(
        self,
        manifold,
        in_features: int,
        out_features: int,
        # 聚合配置
        aggregation_mode: Literal["mean", "attention", "max", "sum"] = "mean",
        aggregation_basepoint: Literal["origin", "nodewise"] = "origin",
        # 激活配置
        activation_type: str = "relu",
        activation_basepoint: Literal["origin", "adaptive"] = "origin",
        # 网络配置
        bias: bool = True,
        dropout: float = 0.0,
        use_self_loops: bool = True,
        # 数值稳定性
        normalize_after_aggregation: bool = True,
        # 残差连接
        residual: bool = False,
        residual_beta: float = 0.5,
        residual_mode: Literal["tangent", "geodesic"] = "geodesic",  # 改进残差模式
        **kwargs
    ):
        # 正确配置MessagePassing - 我们完全自定义聚合
        super().__init__(aggr=None, flow='source_to_target', node_dim=0, **kwargs)

        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.use_self_loops = use_self_loops
        self.normalize_after_aggregation = normalize_after_aggregation
        self.residual = residual and (in_features == out_features)
        self.residual_beta = residual_beta
        self.residual_mode = residual_mode

        # 线性变换层（在切空间执行）
        self.linear = HyperbolicLinear(
            manifold=manifold,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            basepoint="origin"  # 统一使用原点
        )

        # 聚合器
        self.aggregator = create_aggregator(
            manifold=manifold,
            mode=aggregation_mode,
            basepoint=aggregation_basepoint,
            attention_dim=out_features + 1 if aggregation_mode == "attention" else None,
            dropout=dropout
        )

        # 激活函数
        self.activation = create_hyperbolic_activation(
            manifold=manifold,
            activation_type=activation_type,
            basepoint=activation_basepoint
        )

        # Dropout
        if dropout > 0:
            self.dropout = HyperbolicDropout(manifold, p=dropout)
        else:
            self.dropout = None

        # 统计信息
        self.register_buffer('forward_count', torch.tensor(0))
        self.register_buffer('numerical_issues_count', torch.tensor(0))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (N, d_in + 1) 节点特征在双曲空间
            edge_index: (2, E) 边索引
            edge_attr: (E,) 可选边权重
            size: 可选的 (src_size, dst_size)

        Returns:
            (N, d_out + 1) 输出特征在双曲空间
        """
        # 更新统计
        self.forward_count += 1

        # 检查输入健康性
        self._check_input_health(x, "input")

        # 保存残差
        residual_x = x if self.residual else None

        # Step 1: 线性变换（在切空间执行）
        x_transformed = self.linear(x)  # (N, d_out + 1)
        self._check_input_health(x_transformed, "after_linear")

        # Step 2: 添加自环（如果需要）
        if self.use_self_loops:
            edge_index, edge_attr = self._add_self_loops(
                edge_index, edge_attr, num_nodes=x.size(0)
            )

        # Step 3: 邻域聚合
        x_aggregated = self.aggregator(
            x=x_transformed,
            edge_index=edge_index,
            alpha=edge_attr,
            size=x.size(0) if size is None else size[1]
        )  # (N, d_out + 1)
        self._check_input_health(x_aggregated, "after_aggregation")

        # Step 4: 超曲面归一化（如果启用）
        if self.normalize_after_aggregation:
            x_aggregated = self._normalize_to_hyperboloid(x_aggregated)

        # Step 5: 激活函数
        x_activated = self.activation(x_aggregated)  # (N, d_out + 1)
        self._check_input_health(x_activated, "after_activation")

        # Step 6: Dropout
        if self.dropout is not None and self.training:
            x_activated = self.dropout(x_activated)

        # Step 7: 残差连接（在切空间执行）
        if self.residual and residual_x is not None:
            x_activated = self._apply_residual_connection(residual_x, x_activated)

        # Step 8: 最终归一化
        output = self._normalize_to_hyperboloid(x_activated)
        self._check_input_health(output, "output")

        return output

    def _add_self_loops(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        num_nodes: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """添加自环"""
        # 创建自环索引
        self_loop_index = torch.arange(num_nodes, device=edge_index.device)
        self_loop_edge_index = torch.stack([self_loop_index, self_loop_index], dim=0)

        # 合并边索引
        edge_index = torch.cat([edge_index, self_loop_edge_index], dim=1)

        # 处理边权重
        if edge_attr is not None:
            # 自环权重设为1
            self_loop_attr = torch.ones(num_nodes, device=edge_attr.device, dtype=edge_attr.dtype)
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        return edge_index, edge_attr

    def _normalize_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """
        超曲面归一化: 确保 <x,x>_L = -1/c 且 x_0 > 0
        """
        return self.manifold.proj(x)

    def _apply_residual_connection(
        self,
        residual: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        改进的几何兼容残差连接
        支持两种模式：切空间线性组合和测地线插值
        """
        if self.residual_mode == "tangent":
            # 原始切空间线性组合方式
            c = self.manifold.c
            origin = torch.zeros_like(x)
            origin[..., 0] = 1.0 / torch.sqrt(c)

            # 转换到切空间
            x_tangent = self.manifold.log(origin, x)
            residual_tangent = self.manifold.log(origin, residual)

            # 加权组合
            combined_tangent = (
                self.residual_beta * x_tangent +
                (1.0 - self.residual_beta) * residual_tangent
            )

            # 映射回双曲空间
            return self.manifold.exp(origin, combined_tangent)

        elif self.residual_mode == "geodesic":
            # 几何上更正确的测地线插值
            return self._geodesic_interpolation(residual, x, self.residual_beta)

        else:
            raise ValueError(f"Unknown residual mode: {self.residual_mode}")

    def _geodesic_interpolation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: float
    ) -> torch.Tensor:
        """
        双曲空间中的测地线插值
        γ(t) = exp_x(t * log_x(y))，其中 t ∈ [0, 1]
        """
        if t == 0.0:
            return x.clone()
        elif t == 1.0:
            return y.clone()
        else:
            # 计算从x到y的对数映射
            log_xy = self.manifold.log(x, y)

            # 缩放切向量
            scaled_tangent = t * log_xy

            # 指数映射回双曲空间
            return self.manifold.exp(x, scaled_tangent)

    def _check_input_health(self, x: torch.Tensor, stage: str):
        """检查数值健康性"""
        if torch.isnan(x).any() or torch.isinf(x).any():
            self.numerical_issues_count += 1
            logger.warning(f"Numerical issues detected at {stage}: "
                         f"NaN: {torch.isnan(x).sum()}, "
                         f"Inf: {torch.isinf(x).sum()}")

        # 检查双曲约束（仅在调试模式下）
        if logger.isEnabledFor(logging.DEBUG):
            constraint_violation = torch.abs(
                self.manifold.dot(x, x) + 1.0 / self.manifold.c
            ).max()
            if constraint_violation > 1e-3:
                logger.debug(f"Hyperboloid constraint violation at {stage}: "
                           f"max error = {constraint_violation:.6f}")

    def get_statistics(self) -> dict:
        """获取层统计信息"""
        stats = {
            'forward_count': int(self.forward_count),
            'numerical_issues_count': int(self.numerical_issues_count),
            'residual_mode': self.residual_mode,
            'residual_enabled': self.residual,
        }

        # 聚合器统计
        if hasattr(self.aggregator, 'get_fallback_ratio'):
            stats['aggregator_fallback_ratio'] = self.aggregator.get_fallback_ratio()

        # 曲率信息
        if hasattr(self.manifold, 'c'):
            stats['curvature'] = float(self.manifold.c)

        return stats

    def reset_statistics(self):
        """重置统计信息"""
        self.forward_count.zero_()
        self.numerical_issues_count.zero_()
        self.aggregator.reset_statistics()

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'aggregation={getattr(self.aggregator, "mode", "unknown")}, '
            f'activation={type(self.activation).__name__}, '
            f'residual={self.residual}({self.residual_mode if self.residual else "N/A"})'
        )


class HGCNBlock(nn.Module):
    """
    HGCN 块：包含多个HGCN层的模块

    支持：
    - 多层堆叠
    - 层间残差连接
    - 逐层曲率学习
    - Dropout和正则化
    """

    def __init__(
        self,
        manifolds,  # 可以是单个manifold或manifold列表（逐层曲率）
        layer_dims: list,  # [input_dim, hidden_dim1, hidden_dim2, ..., output_dim]
        # 层配置
        aggregation_mode: str = "mean",
        activation_type: str = "relu",
        dropout: float = 0.0,
        # 残差配置
        residual: bool = True,
        residual_beta: float = 0.5,
        # 归一化配置
        normalize_between_layers: bool = True,
        **layer_kwargs
    ):
        super().__init__()

        if not isinstance(manifolds, (list, tuple)):
            # 单个manifold，复制到所有层
            self.manifolds = [manifolds] * (len(layer_dims) - 1)
        else:
            self.manifolds = list(manifolds)

        if len(self.manifolds) != len(layer_dims) - 1:
            raise ValueError(f"Number of manifolds ({len(self.manifolds)}) must match "
                           f"number of layers ({len(layer_dims) - 1})")

        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.normalize_between_layers = normalize_between_layers

        # 创建层
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = HGCNLayer(
                manifold=self.manifolds[i],
                in_features=layer_dims[i],
                out_features=layer_dims[i + 1],
                aggregation_mode=aggregation_mode,
                activation_type=activation_type,
                dropout=dropout,
                residual=residual and (layer_dims[i] == layer_dims[i + 1]),
                residual_beta=residual_beta,
                residual_mode=layer_kwargs.get('residual_mode', 'geodesic'),  # 默认使用几何正确的模式
                **layer_kwargs
            )
            self.layers.append(layer)

        # 层间归一化
        if normalize_between_layers:
            self.layer_norms = nn.ModuleList([
                HyperbolicLayerNorm(self.manifolds[i], layer_dims[i + 1])
                for i in range(self.num_layers - 1)  # 最后一层不需要
            ])
        else:
            self.layer_norms = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播"""
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)

            # 层间归一化（除了最后一层）
            if (self.layer_norms is not None and
                i < len(self.layers) - 1):
                x = self.layer_norms[i](x)

        return x

    def get_layer_statistics(self) -> dict:
        """获取所有层的统计信息"""
        stats = {}
        for i, layer in enumerate(self.layers):
            stats[f'layer_{i}'] = layer.get_statistics()
        return stats


class HyperbolicLayerNorm(nn.Module):
    """
    双曲空间的层归一化

    在切空间执行LayerNorm，然后映射回双曲空间
    """

    def __init__(
        self,
        manifold,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        super().__init__()

        self.manifold = manifold
        self.normalized_shape = normalized_shape  # 切空间维度
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (..., d+1) 双曲空间特征

        Returns:
            (..., d+1) 归一化后的双曲空间特征
        """
        # 转换到切空间
        c = self.manifold.c
        origin = torch.zeros_like(x)
        origin[..., 0] = 1.0 / torch.sqrt(c)

        x_tangent = self.manifold.log(origin, x)  # (..., d+1)
        x_tangent_pure = self.manifold.from_tangent_dplus1(x_tangent)  # (..., d)

        # LayerNorm
        normalized = F.layer_norm(
            x_tangent_pure,
            (self.normalized_shape,),
            self.weight,
            self.bias,
            self.eps
        )

        # 转换回双曲空间
        normalized_dplus1 = self.manifold.to_tangent_dplus1(normalized)
        output = self.manifold.exp(origin, normalized_dplus1)

        return output


def create_hgcn_layer(
    manifold,
    in_features: int,
    out_features: int,
    **kwargs
) -> HGCNLayer:
    """创建HGCN层的便捷函数"""
    return HGCNLayer(manifold, in_features, out_features, **kwargs)


def create_hgcn_block(
    manifolds,
    layer_dims: list,
    **kwargs
) -> HGCNBlock:
    """创建HGCN块的便捷函数"""
    return HGCNBlock(manifolds, layer_dims, **kwargs)
