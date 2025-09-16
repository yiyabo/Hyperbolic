"""
双曲激活函数实现

该模块实现了保持双曲约束的激活函数，包括：
- 双曲 ReLU
- 双曲 Tanh
- 双曲 ELU
- 双曲 LeakyReLU
- 自定义双曲激活

核心思想：
σ_H(x) = exp_o^(c)(σ(log_o^(c)(x)))

即：双曲空间 → 切空间 → 欧几里得激活 → 双曲空间
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Callable
import warnings


class HyperbolicActivation(nn.Module):
    """
    双曲激活函数基类

    将欧几里得激活函数扩展到双曲空间：
    1. 通过对数映射到基点的切空间
    2. 在切空间应用欧几里得激活
    3. 通过指数映射回双曲空间
    """

    def __init__(
        self,
        manifold,
        activation_fn: Callable[[torch.Tensor], torch.Tensor],
        basepoint: Literal["origin", "adaptive"] = "origin"
    ):
        super().__init__()
        self.manifold = manifold
        self.activation_fn = activation_fn
        self.basepoint = basepoint

    def _get_basepoint(self, x: torch.Tensor) -> torch.Tensor:
        """获取基点"""
        if self.basepoint == "origin":
            # 使用原点 o = (1/√c, 0, ..., 0)
            c = self.manifold.c
            basepoint = torch.zeros_like(x)
            basepoint[..., 0] = 1.0 / torch.sqrt(c)
            return basepoint
        elif self.basepoint == "adaptive":
            # 使用输入的质心作为自适应基点，而非输入点本身
            # 这样保持了几何意义同时避免恒等变换
            if x.dim() > 1:
                # 计算批次质心
                centroid = x.mean(dim=-2, keepdim=True)  # 沿batch维度求平均
                # 投影回双曲面确保是有效基点
                centroid_proj = self.manifold.proj(centroid)
                return centroid_proj.expand_as(x)
            else:
                # 单个样本情况，退化到原点
                c = self.manifold.c
                basepoint = torch.zeros_like(x)
                basepoint[..., 0] = 1.0 / torch.sqrt(c)
                return basepoint
        else:
            raise ValueError(f"Unknown basepoint mode: {self.basepoint}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (..., d+1) 双曲空间中的点

        Returns:
            (..., d+1) 激活后的双曲空间点
        """
        # 获取基点
        basepoint = self._get_basepoint(x)

        # 对数映射到切空间
        x_tangent = self.manifold.log(basepoint, x)  # (..., d+1)

        # 在切空间应用激活函数
        # 注意：切空间向量的第0维在基点为原点时应该为0
        # 但为了数值稳定性，我们对整个向量应用激活
        x_activated = self.activation_fn(x_tangent)

        # 指数映射回双曲空间
        result = self.manifold.exp(basepoint, x_activated)

        return result


class HyperbolicReLU(HyperbolicActivation):
    """双曲 ReLU 激活函数"""

    def __init__(self, manifold, basepoint: Literal["origin", "adaptive"] = "origin"):
        super().__init__(manifold, F.relu, basepoint)


class HyperbolicLeakyReLU(HyperbolicActivation):
    """双曲 Leaky ReLU 激活函数"""

    def __init__(
        self,
        manifold,
        negative_slope: float = 0.01,
        basepoint: Literal["origin", "adaptive"] = "origin"
    ):
        self.negative_slope = negative_slope
        activation_fn = lambda x: F.leaky_relu(x, negative_slope=negative_slope)
        super().__init__(manifold, activation_fn, basepoint)


class HyperbolicTanh(HyperbolicActivation):
    """双曲 Tanh 激活函数"""

    def __init__(self, manifold, basepoint: Literal["origin", "adaptive"] = "origin"):
        super().__init__(manifold, torch.tanh, basepoint)


class HyperbolicELU(HyperbolicActivation):
    """双曲 ELU 激活函数"""

    def __init__(
        self,
        manifold,
        alpha: float = 1.0,
        basepoint: Literal["origin", "adaptive"] = "origin"
    ):
        self.alpha = alpha
        activation_fn = lambda x: F.elu(x, alpha=alpha)
        super().__init__(manifold, activation_fn, basepoint)


class HyperbolicSigmoid(HyperbolicActivation):
    """双曲 Sigmoid 激活函数"""

    def __init__(self, manifold, basepoint: Literal["origin", "adaptive"] = "origin"):
        super().__init__(manifold, torch.sigmoid, basepoint)


class HyperbolicGELU(HyperbolicActivation):
    """双曲 GELU 激活函数"""

    def __init__(self, manifold, basepoint: Literal["origin", "adaptive"] = "origin"):
        super().__init__(manifold, F.gelu, basepoint)


class HyperbolicSwish(HyperbolicActivation):
    """双曲 Swish 激活函数"""

    def __init__(self, manifold, basepoint: Literal["origin", "adaptive"] = "origin"):
        def swish(x):
            return x * torch.sigmoid(x)
        super().__init__(manifold, swish, basepoint)


class AdaptiveHyperbolicActivation(nn.Module):
    """
    自适应双曲激活函数

    根据输入的"双曲性"自动选择合适的激活策略：
    - 对于接近双曲面中心的点，使用标准双曲激活
    - 对于远离中心的点，使用更保守的激活
    """

    def __init__(
        self,
        manifold,
        activation_type: str = "relu",
        radius_threshold: float = 2.0,
        **activation_kwargs
    ):
        super().__init__()
        self.manifold = manifold
        self.radius_threshold = radius_threshold

        # 创建两种激活函数 - 改进策略选择
        self.conservative_activation = self._create_activation("origin", activation_type, **activation_kwargs)
        self.standard_activation = self._create_activation("origin", activation_type, **activation_kwargs)

        # 为保守激活创建更小的激活强度
        if hasattr(self.conservative_activation, 'activation_fn'):
            # 包装激活函数，降低强度
            original_fn = self.conservative_activation.activation_fn
            self.conservative_activation.activation_fn = lambda x: 0.5 * original_fn(x)

    def _create_activation(self, basepoint: str, activation_type: str, **kwargs):
        """创建激活函数"""
        activation_map = {
            "relu": HyperbolicReLU,
            "leaky_relu": HyperbolicLeakyReLU,
            "tanh": HyperbolicTanh,
            "elu": HyperbolicELU,
            "sigmoid": HyperbolicSigmoid,
            "gelu": HyperbolicGELU,
            "swish": HyperbolicSwish,
        }

        if activation_type not in activation_map:
            raise ValueError(f"Unknown activation type: {activation_type}")

        return activation_map[activation_type](self.manifold, basepoint=basepoint, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (..., d+1) 双曲空间中的点

        Returns:
            (..., d+1) 激活后的双曲空间点
        """
        # 计算到原点的距离
        c = self.manifold.c
        origin = torch.zeros_like(x)
        origin[..., 0] = 1.0 / torch.sqrt(c)

        distances = self.manifold.dist(origin, x)  # (...)

        # 根据距离选择激活策略
        conservative_mask = distances > self.radius_threshold

        if conservative_mask.any() and not conservative_mask.all():
            # 混合情况：需要分别处理
            result = torch.zeros_like(x)

            if conservative_mask.any():
                conservative_x = x[conservative_mask]
                result[conservative_mask] = self.conservative_activation(conservative_x)

            if not conservative_mask.all():
                standard_x = x[~conservative_mask]
                result[~conservative_mask] = self.standard_activation(standard_x)

            return result
        elif conservative_mask.all():
            # 全部使用保守激活
            return self.conservative_activation(x)
        else:
            # 全部使用标准激活
            return self.standard_activation(x)


class HyperbolicIdentity(nn.Module):
    """双曲恒等激活函数（用于消融实验）"""

    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class HyperbolicDropout(nn.Module):
    """
    几何感知的双曲空间Dropout

    避免直接在切空间应用标准dropout，而是使用几何兼容的随机置零策略
    """

    def __init__(self, manifold, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.manifold = manifold
        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        # 几何感知的dropout：随机将一些节点移向原点
        c = self.manifold.c
        origin = torch.zeros_like(x)
        origin[..., 0] = 1.0 / torch.sqrt(c)

        # 生成dropout掩码
        if x.dim() > 1:
            # 批次处理：每个样本独立dropout
            mask_shape = x.shape[:-1]  # 除了最后一个特征维度
        else:
            mask_shape = (1,)

        dropout_mask = torch.rand(mask_shape, device=x.device, dtype=x.dtype) > self.p

        if self.inplace:
            result = x
        else:
            result = x.clone()

        # 对被dropout的节点，进行几何上的"软化"操作
        # 而不是简单的置零，将它们向原点移动
        dropped_indices = ~dropout_mask

        if dropped_indices.any():
            # 计算向原点的测地线插值
            alpha = 0.1  # 向原点移动的程度

            if x.dim() > 1:
                x_dropped = x[dropped_indices]
                origin_expanded = origin[dropped_indices] if origin.shape == x.shape else origin.expand_as(x)[dropped_indices]
            else:
                x_dropped = x.unsqueeze(0)
                origin_expanded = origin.unsqueeze(0)

            # 测地线插值: γ(t) = exp_x(t * log_x(origin))
            log_to_origin = self.manifold.log(x_dropped, origin_expanded)
            scaled_tangent = alpha * log_to_origin
            interpolated = self.manifold.exp(x_dropped, scaled_tangent)

            if x.dim() > 1:
                result[dropped_indices] = interpolated
            else:
                result = interpolated.squeeze(0)

        # 重新归一化以保持期望值
        if not dropped_indices.all():
            scale_factor = 1.0 / (1.0 - self.p)

            # 在切空间进行缩放
            x_tangent = self.manifold.log(origin, result)
            x_tangent_scaled = scale_factor * x_tangent
            result = self.manifold.exp(origin, x_tangent_scaled)

        return result


def create_hyperbolic_activation(
    manifold,
    activation_type: str = "relu",
    basepoint: str = "origin",
    **kwargs
) -> nn.Module:
    """
    创建双曲激活函数的便捷函数

    Args:
        manifold: Lorentz流形对象
        activation_type: 激活函数类型
        basepoint: 基点模式
        **kwargs: 激活函数特定参数

    Returns:
        双曲激活函数实例
    """
    activation_map = {
        "relu": HyperbolicReLU,
        "leaky_relu": HyperbolicLeakyReLU,
        "tanh": HyperbolicTanh,
        "elu": HyperbolicELU,
        "sigmoid": HyperbolicSigmoid,
        "gelu": HyperbolicGELU,
        "swish": HyperbolicSwish,
        "identity": HyperbolicIdentity,
        "adaptive": AdaptiveHyperbolicActivation,
    }

    if activation_type not in activation_map:
        raise ValueError(f"Unknown activation type: {activation_type}. "
                        f"Supported types: {list(activation_map.keys())}")

    if activation_type == "identity":
        return activation_map[activation_type](manifold)
    elif activation_type == "adaptive":
        return activation_map[activation_type](manifold, **kwargs)
    else:
        return activation_map[activation_type](manifold, basepoint=basepoint, **kwargs)