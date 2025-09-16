"""
HGCN (双曲图卷积网络) 主模型实现

该模块实现了完整的HGCN模型，包括：
- 欧几里得输入层（ESM-2 → 双曲空间）
- 多层HGCN卷积
- 双曲距离解码器
- 支持消融实验
- 统计监控
- 可学习曲率

模型流程：
1. ESM-2特征 → 双曲输入层 → 双曲空间
2. 多层HGCN卷积（聚合、线性变换、激活）
3. 双曲距离解码器 → 链路预测分数
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Literal
import logging

try:
    # 相对导入（用于包内导入）
    from ..geometry.manifolds import Lorentz, create_lorentz_manifold
    from .components.linear_layer import HyperbolicInputLayer, HyperbolicOutputLayer
    from .components.hgcn_layer import HGCNLayer, HGCNBlock, create_hgcn_layer
    from .decoders.distance_decoder import HyperbolicDistanceDecoder, create_decoder
except ImportError:
    # 绝对导入（用于脚本直接运行）
    from geometry.manifolds import Lorentz, create_lorentz_manifold
    from models.components.linear_layer import HyperbolicInputLayer, HyperbolicOutputLayer
    from models.components.hgcn_layer import HGCNLayer, HGCNBlock, create_hgcn_layer
    from models.decoders.distance_decoder import HyperbolicDistanceDecoder, create_decoder

logger = logging.getLogger(__name__)


class HGCN(nn.Module):
    """
    HGCN (双曲图卷积网络) 主模型

    完整的蛋白质相互作用预测模型，支持：
    - ESM-2特征输入
    - 可配置的HGCN层
    - 多种解码器选择
    - 消融实验支持
    - 逐层或全局曲率学习
    """

    def __init__(
        self,
        # 输入配置
        input_dim: int,  # ESM-2特征维度
        hidden_dims: List[int],  # 隐藏层维度列表

        # 曲率配置
        curvature: Union[float, List[float]] = 1.0,
        learnable_curvature: bool = True,
        curvature_per_layer: bool = False,

        # HGCN层配置
        aggregation_mode: str = "mean",
        activation_type: str = "relu",
        dropout: float = 0.0,
        use_bias: bool = True,

        # 残差连接
        residual: bool = True,
        residual_beta: float = 0.5,

        # 解码器配置
        decoder_type: str = "distance",
        temperature: Union[float, str] = "learnable",

        # 正则化和稳定性
        input_dropout: float = 0.0,
        normalize_features: bool = True,
        use_self_loops: bool = True,

        **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.curvature_per_layer = curvature_per_layer
        self.decoder_type = decoder_type

        # 构建层维度列表: [input_dim, hidden_dim1, ..., hidden_dimN]
        self.layer_dims = [input_dim] + hidden_dims

        # 创建流形对象
        self.manifolds = self._create_manifolds(
            curvature, learnable_curvature, curvature_per_layer
        )

        # 输入层：ESM-2 → 双曲空间
        self.input_layer = HyperbolicInputLayer(
            manifold=self.manifolds[0],
            in_features=input_dim,
            out_features=hidden_dims[0],
            dropout=input_dropout,
            activation=True
        )

        # HGCN层
        self.hgcn_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):  # 输入层已经处理了第一个变换
            layer = HGCNLayer(
                manifold=self.manifolds[i + 1] if curvature_per_layer else self.manifolds[0],
                in_features=hidden_dims[i],
                out_features=hidden_dims[i + 1],
                aggregation_mode=aggregation_mode,
                activation_type=activation_type,
                dropout=dropout,
                bias=use_bias,
                residual=residual and (hidden_dims[i] == hidden_dims[i + 1]),
                residual_beta=residual_beta,
                use_self_loops=use_self_loops,
                **kwargs
            )
            self.hgcn_layers.append(layer)

        # 解码器
        output_manifold = (self.manifolds[-1] if curvature_per_layer
                          else self.manifolds[0])
        self.decoder = create_decoder(
            manifold=output_manifold,
            decoder_type=decoder_type,
            temperature=temperature,
            **kwargs
        )

        # 特征归一化
        if normalize_features:
            self.feature_norm = nn.LayerNorm(input_dim)
        else:
            self.feature_norm = None

        # 统计信息
        self.register_buffer('forward_count', torch.tensor(0))
        self.register_buffer('edge_prediction_count', torch.tensor(0))

        self._initialize_parameters()

    def _create_manifolds(
        self,
        curvature: Union[float, List[float]],
        learnable: bool,
        per_layer: bool
    ) -> List[Lorentz]:
        """创建流形对象 - 改进边界情况处理"""
        # 输入验证
        if isinstance(curvature, (list, tuple)):
            if len(curvature) == 0:
                raise ValueError("Curvature list cannot be empty")
            # 验证所有曲率值都为正数
            for i, c in enumerate(curvature):
                if not isinstance(c, (int, float)) or c <= 0:
                    raise ValueError(f"Curvature at index {i} must be a positive number, got {c}")
        elif isinstance(curvature, (int, float)):
            if curvature <= 0:
                raise ValueError(f"Curvature must be positive, got {curvature}")
        else:
            raise ValueError(f"Invalid curvature type: {type(curvature)}. "
                           f"Expected float or list of floats.")

        if per_layer:
            if isinstance(curvature, (int, float)):
                # 统一曲率，但每层独立可学习
                manifolds = [
                    create_lorentz_manifold(c=float(curvature), learnable=learnable)
                    for _ in range(self.num_layers)
                ]
            elif isinstance(curvature, (list, tuple)):
                # 每层指定曲率
                if len(curvature) != self.num_layers:
                    raise ValueError(f"Curvature list length ({len(curvature)}) "
                                   f"must match number of layers ({self.num_layers})")
                manifolds = [
                    create_lorentz_manifold(c=float(c), learnable=learnable)
                    for c in curvature
                ]
        else:
            # 全局曲率
            if isinstance(curvature, (list, tuple)):
                if len(curvature) > 1:
                    logger.warning(f"Using only first curvature value {curvature[0]} "
                                 f"for global curvature (ignoring {curvature[1:]})")
                curvature = curvature[0]  # 只使用第一个值
            manifolds = [create_lorentz_manifold(c=float(curvature), learnable=learnable)]

        return nn.ModuleList(manifolds)

    def _initialize_parameters(self):
        """初始化模型参数"""
        # 大部分初始化在各个组件内部完成
        logger.info(f"Initialized HGCN model:")
        logger.info(f"  Input dim: {self.input_dim}")
        logger.info(f"  Hidden dims: {self.hidden_dims}")
        logger.info(f"  Num layers: {self.num_layers}")
        logger.info(f"  Curvature per layer: {self.curvature_per_layer}")
        logger.info(f"  Decoder type: {self.decoder_type}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播（训练时的完整图卷积）

        Args:
            x: (N, input_dim) 节点特征（ESM-2）
            edge_index: (2, E) 边索引
            edge_attr: (E,) 可选边权重
            return_embeddings: 是否返回节点嵌入

        Returns:
            embeddings: (N, hidden_dims[-1] + 1) 节点嵌入在双曲空间
            如果 return_embeddings=False，则不返回
        """
        self.forward_count += 1

        # 输入特征归一化
        if self.feature_norm is not None:
            x = self.feature_norm(x)

        # Step 1: 输入层 ESM-2 → 双曲空间
        h = self.input_layer(x)  # (N, hidden_dims[0] + 1)

        # Step 2: HGCN层
        for i, layer in enumerate(self.hgcn_layers):
            h = layer(h, edge_index, edge_attr)  # (N, hidden_dims[i+1] + 1)

        if return_embeddings:
            return h
        else:
            return h

    def predict_links(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        return_distances: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        链路预测

        Args:
            h: (N, d+1) 节点嵌入在双曲空间
            edge_index: (2, E) 待预测的边
            return_distances: 是否返回距离值

        Returns:
            scores: (E,) 链路预测分数
            distances: (E,) 双曲距离 (如果需要)
        """
        self.edge_prediction_count += 1

        return self.decoder.forward_edges(h, edge_index, return_distances)

    def predict_pairs(
        self,
        h: torch.Tensor,
        pairs: torch.Tensor,
        return_distances: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        节点对预测

        Args:
            h: (N, d+1) 节点嵌入
            pairs: (P, 2) 节点对
            return_distances: 是否返回距离

        Returns:
            scores: (P,) 预测分数
            distances: (P,) 双曲距离 (如果需要)
        """
        return self.decoder.forward_pairs(h, pairs, return_distances)

    def score_all_pairs(
        self,
        h: torch.Tensor,
        chunk_size: Optional[int] = 1000
    ) -> torch.Tensor:
        """
        计算所有节点对的分数（评估用）

        Args:
            h: (N, d+1) 节点嵌入
            chunk_size: 分块大小，避免内存溢出

        Returns:
            scores: (N, N) 分数矩阵
        """
        if hasattr(self.decoder, 'score_all_pairs'):
            return self.decoder.score_all_pairs(h, chunk_size)
        else:
            # 手动实现
            N = h.size(0)
            scores = torch.zeros(N, N, device=h.device, dtype=h.dtype)

            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                pairs = torch.stack([
                    torch.arange(i, end_i, device=h.device).repeat_interleave(N),
                    torch.arange(N, device=h.device).repeat(end_i - i)
                ], dim=1)

                chunk_scores = self.predict_pairs(h, pairs)
                scores[i:end_i] = chunk_scores.view(end_i - i, N)

            return scores

    def get_curvatures(self) -> Dict[str, float]:
        """获取所有流形的曲率参数"""
        curvatures = {}

        if self.curvature_per_layer:
            for i, manifold in enumerate(self.manifolds):
                curvatures[f'layer_{i}'] = float(manifold.c)
        else:
            curvatures['global'] = float(self.manifolds[0].c)

        return curvatures

    def get_statistics(self) -> Dict:
        """获取模型统计信息 - 改进多进程支持和错误处理"""
        try:
            stats = {
                'forward_count': int(self.forward_count.item() if hasattr(self.forward_count, 'item')
                                   else self.forward_count),
                'edge_prediction_count': int(self.edge_prediction_count.item() if hasattr(self.edge_prediction_count, 'item')
                                           else self.edge_prediction_count),
                'curvatures': self.get_curvatures(),
                'model_info': {
                    'input_dim': self.input_dim,
                    'hidden_dims': self.hidden_dims,
                    'num_layers': self.num_layers,
                    'curvature_per_layer': self.curvature_per_layer,
                    'decoder_type': self.decoder_type,
                }
            }

            # 解码器统计 - 安全获取
            try:
                if hasattr(self.decoder, 'get_statistics'):
                    stats['decoder_stats'] = self.decoder.get_statistics()
                else:
                    stats['decoder_stats'] = {'type': type(self.decoder).__name__}
            except Exception as e:
                logger.warning(f"Failed to get decoder statistics: {e}")
                stats['decoder_stats'] = {'error': str(e)}

            # 层级统计 - 安全获取
            layer_stats = {}
            try:
                if hasattr(self.input_layer, 'get_statistics'):
                    layer_stats['input_layer'] = self.input_layer.get_statistics()
            except Exception as e:
                logger.warning(f"Failed to get input layer statistics: {e}")
                layer_stats['input_layer'] = {'error': str(e)}

            for i, layer in enumerate(self.hgcn_layers):
                try:
                    if hasattr(layer, 'get_statistics'):
                        layer_stats[f'hgcn_layer_{i}'] = layer.get_statistics()
                    else:
                        layer_stats[f'hgcn_layer_{i}'] = {'type': type(layer).__name__}
                except Exception as e:
                    logger.warning(f"Failed to get layer {i} statistics: {e}")
                    layer_stats[f'hgcn_layer_{i}'] = {'error': str(e)}

            if layer_stats:
                stats['layer_stats'] = layer_stats

            return stats

        except Exception as e:
            logger.error(f"Failed to get model statistics: {e}")
            return {
                'error': str(e),
                'forward_count': 0,
                'edge_prediction_count': 0,
                'curvatures': {},
                'model_info': {
                    'input_dim': getattr(self, 'input_dim', -1),
                    'hidden_dims': getattr(self, 'hidden_dims', []),
                    'num_layers': getattr(self, 'num_layers', -1),
                }
            }

    def reset_statistics(self):
        """重置统计信息"""
        self.forward_count.zero_()
        self.edge_prediction_count.zero_()

        # 重置各层统计
        if hasattr(self.decoder, 'reset_statistics'):
            self.decoder.reset_statistics()

        for layer in self.hgcn_layers:
            if hasattr(layer, 'reset_statistics'):
                layer.reset_statistics()

    def get_memory_usage(self) -> Dict[str, int]:
        """获取内存使用情况 - 改进参数统计和内存估算"""
        def get_param_count(module):
            return sum(p.numel() for p in module.parameters())

        def get_param_memory(module):
            """估算参数内存使用（字节）"""
            total_elements = sum(p.numel() for p in module.parameters())
            # 假设float32，每个参数4字节
            return total_elements * 4

        try:
            memory_info = {
                'total_params': get_param_count(self),
                'total_param_memory_mb': get_param_memory(self) / (1024 * 1024),
                'input_layer_params': get_param_count(self.input_layer),
                'hgcn_layers_params': sum(get_param_count(layer) for layer in self.hgcn_layers),
                'decoder_params': get_param_count(self.decoder),
                'manifolds_params': sum(get_param_count(manifold) for manifold in self.manifolds),
            }

            # 详细的层级参数统计
            layer_params = {}
            for i, layer in enumerate(self.hgcn_layers):
                layer_params[f'layer_{i}'] = get_param_count(layer)
            memory_info['layer_params_detail'] = layer_params

            # 可学习曲率参数统计
            learnable_curvatures = sum(1 for m in self.manifolds if m.learnable)
            memory_info['learnable_curvature_params'] = learnable_curvatures

            return memory_info

        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {'error': str(e), 'total_params': -1}

    def set_curvature_requires_grad(self, requires_grad: bool):
        """设置曲率参数是否需要梯度"""
        for manifold in self.manifolds:
            if manifold.learnable:
                manifold.c_param.requires_grad_(requires_grad)

    def clamp_curvatures(self, min_c: float = 1e-4, max_c: float = 10.0):
        """限制曲率参数范围"""
        with torch.no_grad():
            for manifold in self.manifolds:
                if manifold.learnable:
                    # 通过softplus的逆函数来限制
                    current_c = manifold.c
                    clamped_c = torch.clamp(current_c, min=min_c, max=max_c)

                    if not torch.equal(current_c, clamped_c):
                        # 更新参数
                        new_param = torch.log(torch.expm1(clamped_c - manifold.c_min))
                        manifold.c_param.copy_(new_param)

    def extra_repr(self) -> str:
        return (
            f'input_dim={self.input_dim}, '
            f'hidden_dims={self.hidden_dims}, '
            f'curvature_per_layer={self.curvature_per_layer}, '
            f'decoder_type={self.decoder_type}'
        )


def create_hgcn_from_config(config: Dict) -> HGCN:
    """
    从配置字典创建HGCN模型 - 改进配置验证和错误处理

    Args:
        config: 模型配置字典

    Returns:
        HGCN模型实例

    Raises:
        ValueError: 配置参数无效
        KeyError: 必需配置缺失
    """
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")

    model_config = config.get('model', {})
    if not isinstance(model_config, dict):
        raise ValueError(f"model config must be a dictionary, got {type(model_config)}")

    # 基本配置 - 添加验证
    input_dim = model_config.get('input_dim', 1280)  # ESM-2 650M维度
    if not isinstance(input_dim, int) or input_dim <= 0:
        raise ValueError(f"input_dim must be a positive integer, got {input_dim}")

    hidden_dims = model_config.get('hidden_dims', [256, 128])
    if not isinstance(hidden_dims, (list, tuple)) or len(hidden_dims) == 0:
        raise ValueError(f"hidden_dims must be a non-empty list, got {hidden_dims}")
    for i, dim in enumerate(hidden_dims):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"hidden_dims[{i}] must be a positive integer, got {dim}")

    # 曲率配置 - 添加验证
    curvature = model_config.get('curvature', 1.0)
    learnable_curvature = model_config.get('learnable_curvature', True)
    curvature_per_layer = model_config.get('curvature_per_layer', False)

    if not isinstance(learnable_curvature, bool):
        raise ValueError(f"learnable_curvature must be boolean, got {learnable_curvature}")
    if not isinstance(curvature_per_layer, bool):
        raise ValueError(f"curvature_per_layer must be boolean, got {curvature_per_layer}")

    # 层配置 - 添加验证
    aggregation_mode = model_config.get('aggregation_mode', 'mean')
    valid_aggregation_modes = ['mean', 'attention', 'max', 'sum']
    if aggregation_mode not in valid_aggregation_modes:
        raise ValueError(f"aggregation_mode must be one of {valid_aggregation_modes}, got {aggregation_mode}")

    activation_type = model_config.get('activation_type', 'relu')
    valid_activation_types = ['relu', 'leaky_relu', 'tanh', 'elu', 'sigmoid', 'gelu', 'swish', 'identity']
    if activation_type not in valid_activation_types:
        raise ValueError(f"activation_type must be one of {valid_activation_types}, got {activation_type}")

    dropout = model_config.get('dropout', 0.1)
    if not isinstance(dropout, (int, float)) or dropout < 0 or dropout >= 1:
        raise ValueError(f"dropout must be a float in [0, 1), got {dropout}")

    # 解码器配置 - 添加验证
    decoder_config = model_config.get('decoder', {})
    if not isinstance(decoder_config, dict):
        decoder_config = {}

    decoder_type = decoder_config.get('type', 'distance')
    valid_decoder_types = ['distance', 'bilinear', 'dot_product']
    if decoder_type not in valid_decoder_types:
        raise ValueError(f"decoder type must be one of {valid_decoder_types}, got {decoder_type}")

    temperature = decoder_config.get('temperature', 'learnable')

    # 其他配置 - 添加验证
    residual = model_config.get('residual', True)
    if not isinstance(residual, bool):
        raise ValueError(f"residual must be boolean, got {residual}")

    normalize_features = model_config.get('normalize_features', True)
    if not isinstance(normalize_features, bool):
        raise ValueError(f"normalize_features must be boolean, got {normalize_features}")

    try:
        return HGCN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            curvature=curvature,
            learnable_curvature=learnable_curvature,
            curvature_per_layer=curvature_per_layer,
            aggregation_mode=aggregation_mode,
            activation_type=activation_type,
            dropout=dropout,
            decoder_type=decoder_type,
            temperature=temperature,
            residual=residual,
            normalize_features=normalize_features,
        )
    except Exception as e:
        raise ValueError(f"Failed to create HGCN model: {e}") from e


# 便捷创建函数
def create_hgcn(
    input_dim: int = 1280,
    hidden_dims: List[int] = None,
    **kwargs
) -> HGCN:
    """创建HGCN模型的便捷函数"""
    if hidden_dims is None:
        hidden_dims = [256, 128]

    return HGCN(input_dim=input_dim, hidden_dims=hidden_dims, **kwargs)
