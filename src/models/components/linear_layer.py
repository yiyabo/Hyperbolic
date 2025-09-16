"""
双曲线性层实现

该模块实现了双曲空间中的线性变换，包括：
- 双曲线性层
- 维度转换处理
- 基点选择策略
- 数值稳定性保证

核心思想：
在统一基点的切空间中执行欧几里得线性变换，然后映射回双曲空间：
log_o^(c)(x) → 线性变换 → exp_o^(c)(结果)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
import warnings


class HyperbolicLinear(nn.Module):
    """
    双曲线性层

    在切空间中执行线性变换：
    1. 输入从双曲空间映射到基点的切空间
    2. 在切空间执行 Wx + b
    3. 结果映射回双曲空间

    维度处理：
    - 输入: (..., d_in + 1) 在双曲空间
    - 切空间: (..., d_in) 去掉第0维
    - 线性变换: d_in → d_out
    - 输出: (..., d_out + 1) 在双曲空间
    """

    def __init__(
        self,
        manifold,
        in_features: int,
        out_features: int,
        bias: bool = True,
        basepoint: Literal["origin", "learnable"] = "origin",
        use_bias_in_tangent: bool = True
    ):
        super().__init__()

        self.manifold = manifold
        self.in_features = in_features  # 切空间维度
        self.out_features = out_features  # 切空间维度
        self.basepoint_mode = basepoint
        self.use_bias_in_tangent = use_bias_in_tangent

        # 线性变换参数（在切空间中操作）
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            if use_bias_in_tangent:
                # bias 在切空间中
                self.bias = nn.Parameter(torch.empty(out_features))
            else:
                # bias 在双曲空间中（作为偏移点）
                self.bias = nn.Parameter(torch.empty(out_features + 1))
        else:
            self.register_parameter('bias', None)

        # 可学习基点
        if basepoint == "learnable":
            self.basepoint = nn.Parameter(torch.empty(in_features + 1))
        else:
            self.register_parameter('basepoint', None)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        # Xavier/Glorot 初始化
        nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            if self.use_bias_in_tangent:
                # 切空间bias初始化为小值
                nn.init.uniform_(self.bias, -0.001, 0.001)
            else:
                # 双曲空间bias初始化为接近原点
                # 延迟到first forward时初始化，避免硬编码曲率
                pass

        if self.basepoint is not None:
            # 延迟到first forward时初始化，避免硬编码曲率
            pass

    def _get_basepoint(self, x: torch.Tensor) -> torch.Tensor:
        """获取基点"""
        if self.basepoint_mode == "origin":
            # 使用原点作为基点
            c = self.manifold.c
            basepoint = torch.zeros_like(x)
            basepoint[..., 0] = 1.0 / torch.sqrt(c)
            return basepoint

        elif self.basepoint_mode == "learnable":
            # 使用可学习基点
            if self.basepoint is None:
                raise RuntimeError("Learnable basepoint not initialized")

            # 首次使用时初始化基点
            if not hasattr(self, '_basepoint_initialized'):
                with torch.no_grad():
                    c = self.manifold.c
                    self.basepoint.zero_()
                    self.basepoint[0] = 1.0 / torch.sqrt(c)
                    # 添加小的随机扰动到其他维度
                    if len(self.basepoint) > 1:
                        self.basepoint[1:].uniform_(-0.01, 0.01)
                    self._basepoint_initialized = True

            # 确保可学习基点在双曲面上
            basepoint = self.manifold.proj(self.basepoint)

            # 广播到输入形状
            input_shape = x.shape[:-1] + (x.shape[-1],)
            basepoint = basepoint.expand(input_shape)

            return basepoint
        else:
            raise ValueError(f"Unknown basepoint mode: {self.basepoint_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (..., d_in + 1) 输入特征在双曲空间

        Returns:
            (..., d_out + 1) 输出特征在双曲空间
        """
        # 检查输入维度
        if x.size(-1) != self.in_features + 1:
            raise ValueError(f"Expected input size (..., {self.in_features + 1}), "
                           f"got (..., {x.size(-1)})")

        # 获取基点
        basepoint = self._get_basepoint(x)

        # Step 1: 映射到切空间
        x_tangent = self.manifold.log(basepoint, x)  # (..., d_in + 1)

        # Step 2: 转换为纯切空间表示（去掉第0维）
        x_tangent_pure = self.manifold.from_tangent_dplus1(x_tangent)  # (..., d_in)

        # Step 3: 线性变换
        output_tangent_pure = F.linear(x_tangent_pure, self.weight)  # (..., d_out)

        # Step 4: 添加切空间bias（如果使用）
        if self.bias is not None and self.use_bias_in_tangent:
            output_tangent_pure = output_tangent_pure + self.bias

        # Step 5: 转换回ambient切空间表示
        output_tangent = self.manifold.to_tangent_dplus1(output_tangent_pure)  # (..., d_out + 1)

        # Step 6: 确定输出基点（改进维度处理）
        c = self.manifold.c
        if self.out_features == self.in_features:
            # 维度不变，使用相同基点
            output_basepoint = basepoint
        else:
            # 维度改变，创建适当维度的基点
            # 确保形状正确：(..., out_features + 1)
            batch_shape = x.shape[:-1]  # 去掉最后的特征维度
            output_basepoint = torch.zeros(batch_shape + (self.out_features + 1,),
                                         device=x.device, dtype=x.dtype)
            output_basepoint[..., 0] = 1.0 / torch.sqrt(c)

            # 如果使用可学习基点且输出维度不同，需要特殊处理
            if self.basepoint_mode == "learnable":
                # 将可学习基点的信息传递到输出基点
                # 这里简化为使用标准原点，但保留可学习性质的影响
                pass

        # Step 7: 映射回双曲空间
        output = self.manifold.exp(output_basepoint, output_tangent)

        # Step 8: 添加双曲空间bias（如果使用）
        if self.bias is not None and not self.use_bias_in_tangent:
            # 首次使用时初始化双曲空间bias
            if not hasattr(self, '_bias_initialized'):
                with torch.no_grad():
                    c = self.manifold.c
                    self.bias.zero_()
                    self.bias[0] = 1.0 / torch.sqrt(c)
                    if len(self.bias) > 1:
                        self.bias[1:].uniform_(-0.001, 0.001)
                    self._bias_initialized = True

            # 双曲空间中的"平移"操作比较复杂，这里简化处理
            # 实际上应该使用双曲平移，但为简化我们在切空间处理
            warnings.warn("Hyperbolic space bias is experimental and may not preserve geometric properties")

            bias_point = self.manifold.proj(self.bias)

            # 创建与输出维度匹配的bias基点
            bias_basepoint = torch.zeros_like(output[..., :1])  # 取第一个样本来确定形状
            bias_basepoint = bias_basepoint.expand(output.shape[:-1] + (output.shape[-1],))
            bias_basepoint[..., 0] = 1.0 / torch.sqrt(self.manifold.c)

            # 将输出和bias都映射到切空间相加
            output_tan = self.manifold.log(bias_basepoint, output)
            bias_expanded = bias_point.unsqueeze(0).expand_as(output)
            bias_tan = self.manifold.log(bias_basepoint, bias_expanded)

            combined_tan = output_tan + bias_tan
            output = self.manifold.exp(bias_basepoint, combined_tan)

        return output

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, basepoint={self.basepoint_mode}')


class HyperbolicInputLayer(nn.Module):
    """
    双曲输入层：欧几里得 → 双曲空间

    专门用于将欧几里得特征（如ESM-2）映射到双曲空间
    """

    def __init__(
        self,
        manifold,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.use_activation = activation

        # 欧几里得预处理
        self.euclidean_transform = nn.Linear(in_features, out_features, bias=bias)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # 可选的预激活
        if activation:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.euclidean_transform.weight)
        if self.euclidean_transform.bias is not None:
            nn.init.zeros_(self.euclidean_transform.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：欧几里得 → 双曲

        Args:
            x: (..., d_in) 欧几里得特征

        Returns:
            (..., d_out + 1) 双曲空间特征
        """
        # 欧几里得预处理
        x_transformed = self.euclidean_transform(x)  # (..., d_out)

        if self.dropout is not None:
            x_transformed = self.dropout(x_transformed)

        if self.activation is not None:
            x_transformed = self.activation(x_transformed)

        # 转换为切空间表示
        x_tangent_dplus1 = self.manifold.to_tangent_dplus1(x_transformed)  # (..., d_out + 1)

        # 基点
        c = self.manifold.c
        basepoint = torch.zeros_like(x_tangent_dplus1)
        basepoint[..., 0] = 1.0 / torch.sqrt(c)

        # 映射到双曲空间
        x_hyperbolic = self.manifold.exp(basepoint, x_tangent_dplus1)

        return x_hyperbolic


class HyperbolicOutputLayer(nn.Module):
    """
    双曲输出层：双曲空间 → 欧几里得

    用于将双曲特征映射回欧几里得空间进行最终处理
    """

    def __init__(
        self,
        manifold,
        in_features: int,
        out_features: int,
        bias: bool = True,
        basepoint: Literal["origin", "mean"] = "origin"
    ):
        super().__init__()

        self.manifold = manifold
        self.in_features = in_features  # 切空间维度
        self.out_features = out_features
        self.basepoint_mode = basepoint

        # 欧几里得后处理
        self.euclidean_transform = nn.Linear(in_features, out_features, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.euclidean_transform.weight)
        if self.euclidean_transform.bias is not None:
            nn.init.zeros_(self.euclidean_transform.bias)

    def _get_basepoint(self, x: torch.Tensor) -> torch.Tensor:
        """获取基点"""
        if self.basepoint_mode == "origin":
            c = self.manifold.c
            basepoint = torch.zeros_like(x)
            basepoint[..., 0] = 1.0 / torch.sqrt(c)
            return basepoint
        elif self.basepoint_mode == "mean":
            # 使用输入的平均点作为基点（简化处理）
            return x.mean(dim=-2, keepdim=True).expand_as(x)
        else:
            raise ValueError(f"Unknown basepoint mode: {self.basepoint_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：双曲 → 欧几里得

        Args:
            x: (..., d_in + 1) 双曲空间特征

        Returns:
            (..., d_out) 欧几里得特征
        """
        # 获取基点
        basepoint = self._get_basepoint(x)

        # 映射到切空间
        x_tangent = self.manifold.log(basepoint, x)  # (..., d_in + 1)

        # 转换为纯切空间表示
        x_tangent_pure = self.manifold.from_tangent_dplus1(x_tangent)  # (..., d_in)

        # 欧几里得变换
        output = self.euclidean_transform(x_tangent_pure)  # (..., d_out)

        return output


class HyperbolicMLPBlock(nn.Module):
    """
    双曲MLP块

    包含线性变换、激活函数、dropout的完整块
    """

    def __init__(
        self,
        manifold,
        in_features: int,
        out_features: int,
        activation_type: str = "relu",
        dropout: float = 0.0,
        bias: bool = True,
        residual: bool = False
    ):
        super().__init__()

        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.residual = residual and (in_features == out_features)

        # 线性层
        self.linear = HyperbolicLinear(manifold, in_features, out_features, bias=bias)

        # 激活函数
        from .activations import create_hyperbolic_activation
        self.activation = create_hyperbolic_activation(manifold, activation_type)

        # Dropout
        if dropout > 0:
            from .activations import HyperbolicDropout
            self.dropout = HyperbolicDropout(manifold, p=dropout)
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        identity = x if self.residual else None

        # 线性变换
        x = self.linear(x)

        # 激活
        x = self.activation(x)

        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # 残差连接（在切空间进行）
        if self.residual and identity is not None:
            # 切空间残差
            c = self.manifold.c
            origin = torch.zeros_like(x)
            origin[..., 0] = 1.0 / torch.sqrt(c)

            x_tan = self.manifold.log(origin, x)
            identity_tan = self.manifold.log(origin, identity)

            # 简单加法
            residual_tan = x_tan + identity_tan
            x = self.manifold.exp(origin, residual_tan)

        return x


def create_hyperbolic_linear(
    manifold,
    in_features: int,
    out_features: int,
    **kwargs
) -> HyperbolicLinear:
    """创建双曲线性层的便捷函数"""
    return HyperbolicLinear(manifold, in_features, out_features, **kwargs)
