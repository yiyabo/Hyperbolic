"""
Lorentz 流形几何操作实现

该模块实现了 Lorentz 模型中的核心几何操作，包括：
- Lorentz 内积和范数
- 双曲距离计算
- 指数/对数映射
- 平行传输
- 投影和归一化
- 聚合操作

数学基础：
- Lorentz 流形: H^d = {x ∈ R^{d+1} : <x,x>_L = -1/c, x_0 > 0}
- Lorentz 内积: <x,y>_L = -x_0*y_0 + Σx_i*y_i
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
import warnings

# 数值稳定性常量
SMALL_GEODESIC = 1e-3  # 小步长阈值
MIN_NORM = 1e-12       # 最小范数（提高到更实用的值）
EPS_CLAMP = 1e-6       # clamp epsilon（提高稳定性）
ACOSH_TAYLOR_THRESH = 0.1  # arccosh泰勒展开阈值


class Lorentz(nn.Module):
    """
    Lorentz 流形实现

    Args:
        c: 曲率参数，c > 0
        learnable: 是否可学习曲率
    """

    def __init__(self, c: float = 1.0, learnable: bool = False):
        super().__init__()
        if learnable:
            # 使用 softplus + c_min 确保 c > 0
            self.c_param = nn.Parameter(torch.tensor(math.log(math.exp(c - 1e-4) - 1)))
            self.c_min = 1e-4
        else:
            self.register_buffer('c_param', torch.tensor(c))
            self.c_min = 0.0

        self.learnable = learnable

    @property
    def c(self) -> torch.Tensor:
        """获取当前曲率参数"""
        if self.learnable:
            return torch.nn.functional.softplus(self.c_param) + self.c_min
        return self.c_param

    def dot(self, x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """
        Lorentz 内积: <x,y>_L = -x_0*y_0 + Σx_i*y_i

        Args:
            x, y: (..., d+1) 张量
            keepdim: 是否保持维度

        Returns:
            (...,) 或 (..., 1) 内积结果
        """
        # Lorentz 内积：第0维是负的，其余是正的
        prod = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)

        if keepdim:
            prod = prod.unsqueeze(-1)

        return prod

    def norm_tan(self, v: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """
        切向量的 Lorentz 范数: ||v||_L = √<v,v>_L

        Args:
            v: (..., d+1) 切向量
            keepdim: 是否保持维度

        Returns:
            (...,) 或 (..., 1) 范数
        """
        norm_sq = self.dot(v, v, keepdim=keepdim)
        # 切向量范数应该为正
        norm_sq = torch.clamp(norm_sq, min=MIN_NORM)
        return torch.sqrt(norm_sq)

    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """
        投影到双曲面: <x,x>_L = -1/c 且 x_0 > 0

        Args:
            x: (..., d+1) 待投影张量

        Returns:
            (..., d+1) 投影后张量
        """
        c = self.c

        # 处理极小的向量 - 映射到原点
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        very_small_mask = x_norm < MIN_NORM

        if very_small_mask.any():
            result = torch.zeros_like(x)
            result[..., 0] = 1.0 / torch.sqrt(c)

            normal_mask = ~very_small_mask.squeeze(-1)
            if normal_mask.any():
                x_normal = x[normal_mask]
                result[normal_mask] = self._proj_single_batch(x_normal, c)

            return result
        else:
            return self._proj_single_batch(x, c)

    def _proj_single_batch(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """对单个批次执行投影，假设输入不为零向量"""
        # 更稳定的投影方法：确保第0个分量为正，然后规范化空间分量

        # 1. 确保第0个分量为正
        x = x.clone()
        negative_mask = x[..., 0] < 0
        x[negative_mask] = -x[negative_mask]

        # 2. 设置第0个分量以满足约束
        # 对于 <x,x>_L = -1/c，有 -x_0^2 + ||x_1:d||^2 = -1/c
        # 所以 x_0 = sqrt(1/c + ||x_1:d||^2)
        spatial_norm_sq = torch.sum(x[..., 1:] ** 2, dim=-1, keepdim=True)
        x0_target = torch.sqrt(1.0 / c + spatial_norm_sq)

        # 3. 构建投影结果
        x_proj = x.clone()
        x_proj[..., 0:1] = x0_target

        return x_proj

    def _acosh_stable(self, z: torch.Tensor) -> torch.Tensor:
        """
        数值稳定的 arccosh 计算

        使用对数形式避免数值不稳定：
        arccosh(z) = log(z + sqrt(z^2 - 1))

        当 z 接近 1 时使用泰勒展开：
        arccosh(1+x) ≈ sqrt(2x) - x/(2*sqrt(2x)) + ...
        """
        z = torch.clamp(z, min=1.0 + EPS_CLAMP)

        # 对于接近1的值使用泰勒展开
        close_to_one = (z - 1.0) < 0.1

        if close_to_one.any():
            result = torch.zeros_like(z)

            # 远离1的情况使用标准公式
            far_mask = ~close_to_one
            if far_mask.any():
                z_far = z[far_mask]
                sqrt_term = torch.sqrt(z_far * z_far - 1.0)
                result[far_mask] = torch.log(z_far + sqrt_term)

            # 接近1的情况使用泰勒展开
            if close_to_one.any():
                x = z[close_to_one] - 1.0
                # arccosh(1+x) ≈ sqrt(2x) * (1 - x/12 + 3*x^2/160)
                sqrt_2x = torch.sqrt(2.0 * x)
                taylor_correction = 1.0 - x/12.0 + 3.0*x*x/160.0
                result[close_to_one] = sqrt_2x * taylor_correction

            return result
        else:
            # 所有值都远离1，使用标准公式
            sqrt_term = torch.sqrt(z * z - 1.0)
            return torch.log(z + sqrt_term)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        双曲距离: d_c(x,y) = (1/√c) * arcosh(-c * <x,y>_L)

        Args:
            x, y: (..., d+1) 双曲空间中的点

        Returns:
            (...,) 距离
        """
        c = self.c
        inner_prod = self.dot(x, y)
        z = -c * inner_prod

        return (1.0 / torch.sqrt(c)) * self._acosh_stable(z)

    def exp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        指数映射: exp_x^(c)(v): T_x H → H

        Args:
            x: (..., d+1) 基点
            v: (..., d+1) 切向量

        Returns:
            (..., d+1) 指数映射结果
        """
        c = self.c
        sqrt_c = torch.sqrt(c)

        # 切向量范数
        v_norm = self.norm_tan(v, keepdim=True)

        # 处理零向量情况
        zero_v_mask = (v_norm < MIN_NORM).squeeze(-1)
        if zero_v_mask.all():
            return x.clone()

        sqrt_c_vnorm = sqrt_c * v_norm

        # 更精确的小范数阈值和泰勒展开
        small_mask = (sqrt_c_vnorm < SMALL_GEODESIC).squeeze(-1)

        result = torch.zeros_like(x)

        if small_mask.any():
            # 改进的小范数泰勒展开（更高精度）
            v_norm_sq = v_norm * v_norm
            c_vnorm_sq = c * v_norm_sq

            # cosh(sqrt(c)|v|) ≈ 1 + c|v|²/2 + c²|v|⁴/24
            coef1 = 1 + c_vnorm_sq / 2 + c_vnorm_sq * c_vnorm_sq / 24

            # sinh(sqrt(c)|v|) / (sqrt(c)|v|) ≈ 1 + c|v|²/6 + c²|v|⁴/120
            coef2 = 1 + c_vnorm_sq / 6 + c_vnorm_sq * c_vnorm_sq / 120

            small_result = coef1 * x + coef2 * v
            result[small_mask] = small_result[small_mask]

        if not small_mask.all():
            # 正常情况 - 添加数值稳定性检查
            normal_mask = ~small_mask
            sqrt_c_vnorm_normal = sqrt_c_vnorm[normal_mask]
            v_norm_normal = v_norm[normal_mask]

            # 避免除零
            safe_vnorm = torch.clamp(v_norm_normal, min=MIN_NORM)

            cosh_term = torch.cosh(sqrt_c_vnorm_normal)
            sinh_term = torch.sinh(sqrt_c_vnorm_normal) / (sqrt_c * safe_vnorm)

            normal_result = cosh_term * x[normal_mask] + sinh_term * v[normal_mask]
            result[normal_mask] = normal_result

        # 处理零向量情况
        if zero_v_mask.any():
            result[zero_v_mask] = x[zero_v_mask]

        # 投影回双曲面
        return self.proj(result)

    def log(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        对数映射: log_x^(c)(y): H → T_x H

        Args:
            x: (..., d+1) 基点
            y: (..., d+1) 目标点

        Returns:
            (..., d+1) 对数映射结果（切向量）
        """
        c = self.c
        inner_prod = self.dot(x, y)
        alpha = -c * inner_prod
        alpha = torch.clamp(alpha, min=1.0 + EPS_CLAMP)

        dist = self._acosh_stable(alpha)

        # 避免除零
        denom = torch.clamp(torch.sqrt(alpha * alpha - 1.0), min=MIN_NORM)

        # 计算切向量
        coef = (dist / denom).unsqueeze(-1)
        tangent_part = y + (c * inner_prod).unsqueeze(-1) * x

        v = coef * tangent_part

        # 确保正交性（数值稳定性）
        orthogonal_proj = self.dot(v, x, keepdim=True)
        v = v - orthogonal_proj * x

        return v

    def transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        平行传输: P_{x→y}^(c)(v): T_x H → T_y H

        Args:
            x: (..., d+1) 起始点
            y: (..., d+1) 目标点
            v: (..., d+1) 待传输的切向量

        Returns:
            (..., d+1) 传输后的切向量
        """
        c = self.c
        inner_xy = self.dot(x, y)
        inner_vy = self.dot(v, y, keepdim=True)

        # 检查点是否相同（避免数值问题）
        points_equal = torch.allclose(x, y, atol=1e-6)
        if points_equal:
            return v.clone()

        # 改进的分母稳定性处理
        raw_denom = 1 - c * inner_xy

        # 检查分母是否接近0（测地线端点情况）
        small_denom_mask = torch.abs(raw_denom) < MIN_NORM

        if small_denom_mask.any():
            # 对于接近对心点的情况，使用替代公式
            # 基于 Gyrovector 空间的平行传输
            result = torch.zeros_like(v)

            # 正常情况
            normal_mask = ~small_denom_mask
            if normal_mask.any():
                safe_denom = torch.clamp(raw_denom[normal_mask], min=MIN_NORM)
                coeff = inner_vy[normal_mask] / safe_denom.unsqueeze(-1)
                result[normal_mask] = v[normal_mask] - coeff * (x[normal_mask] + y[normal_mask])

            # 特殊情况：使用对数-指数映射
            if small_denom_mask.any():
                # v_transported = exp_y(P_γ(log_x(v)))，其中P_γ是沿测地线的平行传输
                # 简化为：直接在y的切空间重新表达v
                for idx in small_denom_mask.nonzero(as_tuple=True)[0]:
                    # 将v投影到y的切空间
                    v_single = v[idx:idx+1]  # 保持batch维度
                    y_single = y[idx:idx+1]

                    # 重新正交化
                    orthogonal_component = self.dot(v_single, y_single, keepdim=True)
                    y_norm_sq = self.dot(y_single, y_single, keepdim=True)
                    result[idx] = (v_single - (orthogonal_component / y_norm_sq) * y_single).squeeze(0)
        else:
            # 标准平行传输公式
            safe_denom = torch.clamp(raw_denom, min=MIN_NORM).unsqueeze(-1)
            transported = v - (inner_vy / safe_denom) * (x + y)
            result = transported

        # 强制正交化修正（确保结果在目标点的切空间）
        orthogonal_proj = self.dot(result, y, keepdim=True)
        y_norm_sq = torch.clamp(self.dot(y, y, keepdim=True), min=MIN_NORM)
        result = result - (orthogonal_proj / y_norm_sq) * y

        return result

    def to_tangent_dplus1(self, vec_d: torch.Tensor) -> torch.Tensor:
        """切空间向量 R^d → ambient形式 (0, vec_d)"""
        zeros = torch.zeros_like(vec_d[..., :1])
        return torch.cat([zeros, vec_d], dim=-1)

    def from_tangent_dplus1(self, vec_dplus1: torch.Tensor) -> torch.Tensor:
        """ambient形式切向量 → 切空间向量 R^d"""
        return vec_dplus1[..., 1:]

    def random_point(self, *size: int, device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
        """
        在双曲面上随机采样点

        Args:
            size: 形状参数
            device: 设备
            dtype: 数据类型

        Returns:
            (*size, d+1) 双曲空间中的随机点
        """
        if device is None:
            device = next(self.parameters()).device if self.learnable else torch.device('cpu')
        if dtype is None:
            dtype = next(self.parameters()).dtype if self.learnable else torch.float32

        # 生成随机向量
        x = torch.randn(*size, device=device, dtype=dtype)

        # 投影到双曲面
        return self.proj(x)

    def random_tangent(self, x: torch.Tensor) -> torch.Tensor:
        """
        在点x的切空间中随机采样切向量

        Args:
            x: (..., d+1) 基点

        Returns:
            (..., d+1) 切向量
        """
        # 生成随机向量
        v = torch.randn_like(x)

        # 投影到切空间：v - <v,x>_L/<x,x>_L * x
        inner_vx = self.dot(v, x, keepdim=True)
        inner_xx = self.dot(x, x, keepdim=True)

        v_tangent = v - (inner_vx / inner_xx) * x

        return v_tangent

    def check_point_on_manifold(self, x: torch.Tensor, atol: float = 1e-5) -> torch.Tensor:
        """检查点是否在双曲面上"""
        c = self.c
        constraint = self.dot(x, x) + 1.0 / c
        return torch.abs(constraint) < atol

    def check_vector_in_tangent(self, x: torch.Tensor, v: torch.Tensor, atol: float = 1e-5) -> torch.Tensor:
        """检查向量是否在切空间中"""
        orthogonality = self.dot(v, x)
        return torch.abs(orthogonality) < atol


def create_lorentz_manifold(c: float = 1.0, learnable: bool = False) -> Lorentz:
    """
    创建 Lorentz 流形实例的便捷函数

    Args:
        c: 曲率参数
        learnable: 是否可学习

    Returns:
        Lorentz 流形实例
    """
    return Lorentz(c=c, learnable=learnable)


# 用于向后兼容的别名
LorentzManifold = Lorentz
