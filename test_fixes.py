#!/usr/bin/env python3
"""
测试脚本：验证HGCN项目的关键修复
====================================

该脚本测试我们修复的关键问题：
1. 几何模块的数值稳定性
2. 聚合器的时样性检查和兜底机制
3. 线性层的维度处理
4. HGCN层的残差连接和MessagePassing
5. 激活函数的基点一致性
6. 主模型的统计管理和配置验证

运行方式：
    python test_fixes.py

或者运行特定测试：
    python test_fixes.py --test geometry
"""

import os
import sys
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import traceback

import torch
import torch.nn as nn
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResults:
    """测试结果收集器"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, test_name: str):
        self.passed += 1
        logger.info(f"✅ {test_name} PASSED")

    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append((test_name, error))
        logger.error(f"❌ {test_name} FAILED: {error}")

    def summary(self):
        total = self.passed + self.failed
        logger.info(f"\n{'='*50}")
        logger.info(f"TEST SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total tests: {total}")
        logger.info(f"Passed: {self.passed}")
        logger.info(f"Failed: {self.failed}")

        if self.failed > 0:
            logger.error(f"\nFAILED TESTS:")
            for test_name, error in self.errors:
                logger.error(f"  - {test_name}: {error}")

        return self.failed == 0


def test_geometry_numerical_stability(results: TestResults):
    """测试几何模块的数值稳定性修复"""
    logger.info("Testing geometry module numerical stability...")

    try:
        from geometry.manifolds import Lorentz

        manifold = Lorentz(c=1.0, learnable=False)

        # 测试1：_acosh_stable 边界情况
        try:
            # 测试接近1的值
            z_close = torch.tensor([1.001, 1.01, 1.1])
            result = manifold._acosh_stable(z_close)
            assert torch.all(torch.isfinite(result)), "arccosh should be finite for values close to 1"

            # 测试边界值
            z_boundary = torch.tensor([1.0000001])
            result_boundary = manifold._acosh_stable(z_boundary)
            assert torch.all(torch.isfinite(result_boundary)), "arccosh should handle boundary values"

            results.add_pass("geometry_acosh_stability")
        except Exception as e:
            results.add_fail("geometry_acosh_stability", str(e))

        # 测试2：proj方法处理零向量
        try:
            # 零向量
            zero_vec = torch.zeros(3, 5)  # batch_size=3, dim=5
            projected = manifold.proj(zero_vec)

            # 检查结果是否为有效的双曲点
            constraint = manifold.dot(projected, projected) + 1.0 / manifold.c
            assert torch.allclose(constraint, torch.zeros_like(constraint), atol=1e-4), \
                "Projected zero vectors should satisfy hyperboloid constraint"

            # 检查是否在上片
            assert torch.all(projected[..., 0] > 0), "Projected points should be on upper sheet"

            results.add_pass("geometry_proj_zero_handling")
        except Exception as e:
            results.add_fail("geometry_proj_zero_handling", str(e))

        # 测试3：指数映射的小范数处理
        try:
            batch_size = 4
            dim = 6

            # 创建基点
            x = manifold.random_point(batch_size, dim + 1)

            # 小范数切向量
            v_small = torch.randn(batch_size, dim + 1) * 1e-4
            # 正交化到切空间
            inner_vx = manifold.dot(v_small, x, keepdim=True)
            inner_xx = manifold.dot(x, x, keepdim=True)
            v_small = v_small - (inner_vx / inner_xx) * x

            result_small = manifold.exp(x, v_small)

            # 检查结果是否有效
            assert torch.all(torch.isfinite(result_small)), "exp should handle small tangent vectors"

            # 检查双曲约束
            constraint = manifold.dot(result_small, result_small) + 1.0 / manifold.c
            assert torch.allclose(constraint, torch.zeros_like(constraint), atol=1e-4), \
                "exp result should satisfy hyperboloid constraint"

            results.add_pass("geometry_exp_small_tangent")
        except Exception as e:
            results.add_fail("geometry_exp_small_tangent", str(e))

        # 测试4：平行传输的稳定性
        try:
            batch_size = 3
            dim = 4

            x = manifold.random_point(batch_size, dim + 1)
            y = manifold.random_point(batch_size, dim + 1)
            v = manifold.random_tangent(x)

            transported = manifold.transport(x, y, v)

            # 检查是否在y的切空间
            orthogonality = manifold.dot(transported, y)
            assert torch.allclose(orthogonality, torch.zeros_like(orthogonality), atol=1e-4), \
                "Transported vector should be orthogonal to target point"

            results.add_pass("geometry_transport_stability")
        except Exception as e:
            results.add_fail("geometry_transport_stability", str(e))

    except ImportError as e:
        results.add_fail("geometry_import", f"Failed to import geometry modules: {e}")
    except Exception as e:
        results.add_fail("geometry_general", f"Unexpected error: {e}")


def test_aggregator_fallback_mechanism(results: TestResults):
    """测试聚合器的兜底机制修复"""
    logger.info("Testing aggregator fallback mechanism...")

    try:
        from geometry.manifolds import Lorentz
        from models.components.aggregators import LorentzAggregator

        manifold = Lorentz(c=1.0, learnable=False)
        aggregator = LorentzAggregator(manifold, mode="mean")

        # 测试1：正常聚合不触发兜底
        try:
            num_nodes = 10
            num_edges = 20
            dim = 6

            # 创建正常的特征
            x = manifold.random_point(num_nodes, dim + 1)

            # 创建边
            edge_index = torch.randint(0, num_nodes, (2, num_edges))

            result = aggregator(x, edge_index, size=num_nodes)

            # 检查结果
            assert torch.all(torch.isfinite(result)), "Aggregation result should be finite"

            # 检查双曲约束
            constraint = manifold.dot(result, result) + 1.0 / manifold.c
            assert torch.allclose(constraint, torch.zeros_like(constraint), atol=1e-3), \
                "Aggregated points should satisfy hyperboloid constraint"

            results.add_pass("aggregator_normal_case")
        except Exception as e:
            results.add_fail("aggregator_normal_case", str(e))

        # 测试2：强制触发兜底机制
        try:
            # 创建可能导致非时样结果的特征（通过构造对称的负值特征）
            num_nodes = 5
            dim = 4

            x = torch.zeros(num_nodes, dim + 1)
            x[..., 0] = 1.0 / torch.sqrt(manifold.c)
            x[..., 1:] = torch.randn(num_nodes, dim) * 0.1

            # 投影到双曲面
            x = manifold.proj(x)

            # 创建边：每个节点连接到所有其他节点
            edge_list = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edge_list.append([i, j])
            edge_index = torch.tensor(edge_list).t()

            # 运行聚合
            result = aggregator(x, edge_index, size=num_nodes)

            # 即使触发兜底，结果也应该有效
            assert torch.all(torch.isfinite(result)), "Fallback mechanism should produce finite results"

            # 检查统计信息
            fallback_ratio = aggregator.get_fallback_ratio()
            logger.info(f"Fallback mechanism used in {fallback_ratio:.2%} of aggregations")

            results.add_pass("aggregator_fallback_mechanism")
        except Exception as e:
            results.add_fail("aggregator_fallback_mechanism", str(e))

    except ImportError as e:
        results.add_fail("aggregator_import", f"Failed to import aggregator modules: {e}")
    except Exception as e:
        results.add_fail("aggregator_general", f"Unexpected error: {e}")


def test_linear_layer_dimension_handling(results: TestResults):
    """测试线性层的维度处理修复"""
    logger.info("Testing linear layer dimension handling...")

    try:
        from geometry.manifolds import Lorentz
        from models.components.linear_layer import HyperbolicLinear, HyperbolicInputLayer

        manifold = Lorentz(c=2.0, learnable=True)  # 使用可学习曲率测试初始化

        # 测试1：维度变化的线性层
        try:
            in_features = 128
            out_features = 64
            batch_size = 8

            linear_layer = HyperbolicLinear(
                manifold=manifold,
                in_features=in_features,
                out_features=out_features,
                bias=True,
                basepoint="origin"
            )

            # 创建输入
            x_euclidean = torch.randn(batch_size, in_features)
            input_layer = HyperbolicInputLayer(manifold, in_features, in_features)
            x = input_layer(x_euclidean)  # 转换到双曲空间

            # 前向传播
            output = linear_layer(x)

            # 检查输出维度
            expected_shape = (batch_size, out_features + 1)
            assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

            # 检查双曲约束
            constraint = manifold.dot(output, output) + 1.0 / manifold.c
            assert torch.allclose(constraint, torch.zeros_like(constraint), atol=1e-3), \
                "Linear layer output should satisfy hyperboloid constraint"

            results.add_pass("linear_dimension_change")
        except Exception as e:
            results.add_fail("linear_dimension_change", str(e))

        # 测试2：可学习基点的初始化
        try:
            linear_learnable = HyperbolicLinear(
                manifold=manifold,
                in_features=32,
                out_features=32,
                basepoint="learnable"
            )

            # 第一次前向传播应该触发基点初始化
            x_test = manifold.random_point(4, 33)  # 32 + 1
            output = linear_learnable(x_test)

            # 检查基点是否被初始化
            assert hasattr(linear_learnable, '_basepoint_initialized'), \
                "Learnable basepoint should be initialized after first forward pass"

            # 检查基点是否在双曲面上
            basepoint = linear_learnable.basepoint
            constraint = manifold.dot(basepoint.unsqueeze(0), basepoint.unsqueeze(0)) + 1.0 / manifold.c
            assert torch.allclose(constraint, torch.zeros_like(constraint), atol=1e-3), \
                "Learnable basepoint should satisfy hyperboloid constraint"

            results.add_pass("linear_learnable_basepoint")
        except Exception as e:
            results.add_fail("linear_learnable_basepoint", str(e))

    except ImportError as e:
        results.add_fail("linear_import", f"Failed to import linear layer modules: {e}")
    except Exception as e:
        results.add_fail("linear_general", f"Unexpected error: {e}")


def test_hgcn_layer_improvements(results: TestResults):
    """测试HGCN层的改进"""
    logger.info("Testing HGCN layer improvements...")

    try:
        from geometry.manifolds import Lorentz
        from models.components.hgcn_layer import HGCNLayer
        import torch_geometric

        manifold = Lorentz(c=1.5, learnable=False)

        # 测试1：残差连接的几何正确性
        try:
            in_features = 64
            batch_size = 6
            num_edges = 20

            layer_geodesic = HGCNLayer(
                manifold=manifold,
                in_features=in_features,
                out_features=in_features,  # 相同维度才能残差
                residual=True,
                residual_mode="geodesic"
            )

            layer_tangent = HGCNLayer(
                manifold=manifold,
                in_features=in_features,
                out_features=in_features,
                residual=True,
                residual_mode="tangent"
            )

            # 创建输入
            x = manifold.random_point(batch_size, in_features + 1)
            edge_index = torch.randint(0, batch_size, (2, num_edges))

            # 测试两种残差模式
            output_geodesic = layer_geodesic(x, edge_index)
            output_tangent = layer_tangent(x, edge_index)

            # 两种输出都应该有效
            for output, mode in [(output_geodesic, "geodesic"), (output_tangent, "tangent")]:
                assert torch.all(torch.isfinite(output)), f"Output should be finite for {mode} residual"

                constraint = manifold.dot(output, output) + 1.0 / manifold.c
                assert torch.allclose(constraint, torch.zeros_like(constraint), atol=1e-3), \
                    f"Output should satisfy hyperboloid constraint for {mode} residual"

            results.add_pass("hgcn_residual_modes")
        except Exception as e:
            results.add_fail("hgcn_residual_modes", str(e))

        # 测试2：MessagePassing配置
        try:
            layer = HGCNLayer(
                manifold=manifold,
                in_features=32,
                out_features=48,
                aggregation_mode="mean"
            )

            # 检查MessagePassing的配置
            assert layer.aggr is None, "Should use custom aggregation (aggr=None)"
            assert layer.flow == 'source_to_target', "Should use source_to_target flow"

            results.add_pass("hgcn_message_passing_config")
        except Exception as e:
            results.add_fail("hgcn_message_passing_config", str(e))

        # 测试3：统计信息收集
        try:
            layer = HGCNLayer(
                manifold=manifold,
                in_features=16,
                out_features=16
            )

            # 运行几次前向传播
            for _ in range(3):
                x = manifold.random_point(4, 17)
                edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
                _ = layer(x, edge_index)

            # 获取统计信息
            stats = layer.get_statistics()

            assert 'forward_count' in stats, "Should collect forward count statistics"
            assert stats['forward_count'] == 3, f"Expected 3 forward passes, got {stats['forward_count']}"

            results.add_pass("hgcn_statistics")
        except Exception as e:
            results.add_fail("hgcn_statistics", str(e))

    except ImportError as e:
        results.add_fail("hgcn_import", f"Failed to import HGCN layer modules: {e}")
    except Exception as e:
        results.add_fail("hgcn_general", f"Unexpected error: {e}")


def test_activation_improvements(results: TestResults):
    """测试激活函数的改进"""
    logger.info("Testing activation function improvements...")

    try:
        from geometry.manifolds import Lorentz
        from models.components.activations import (
            HyperbolicReLU, AdaptiveHyperbolicActivation, HyperbolicDropout
        )

        manifold = Lorentz(c=1.0, learnable=False)

        # 测试1：自适应基点的改进
        try:
            relu_adaptive = HyperbolicReLU(manifold, basepoint="adaptive")

            batch_size = 8
            dim = 10
            x = manifold.random_point(batch_size, dim + 1)

            output = relu_adaptive(x)

            # 检查输出有效性
            assert torch.all(torch.isfinite(output)), "Adaptive activation output should be finite"

            constraint = manifold.dot(output, output) + 1.0 / manifold.c
            assert torch.allclose(constraint, torch.zeros_like(constraint), atol=1e-3), \
                "Adaptive activation should preserve hyperboloid constraint"

            results.add_pass("activation_adaptive_basepoint")
        except Exception as e:
            results.add_fail("activation_adaptive_basepoint", str(e))

        # 测试2：AdaptiveHyperbolicActivation的改进策略
        try:
            adaptive_activation = AdaptiveHyperbolicActivation(
                manifold=manifold,
                activation_type="relu",
                radius_threshold=1.0
            )

            # 创建距离原点远近不同的点
            x_close = manifold.random_point(4, 6)  # 接近原点

            # 创建远离原点的点
            c = manifold.c
            origin = torch.zeros(4, 6)
            origin[..., 0] = 1.0 / torch.sqrt(c)

            # 通过大的切向量创建远点
            large_tangent = torch.randn(4, 6) * 2.0
            # 正交化
            inner = manifold.dot(large_tangent, origin, keepdim=True)
            inner_oo = manifold.dot(origin, origin, keepdim=True)
            large_tangent = large_tangent - (inner / inner_oo) * origin

            x_far = manifold.exp(origin, large_tangent)

            # 测试自适应激活
            output_close = adaptive_activation(x_close)
            output_far = adaptive_activation(x_far)

            # 两种输出都应该有效
            for output, desc in [(output_close, "close"), (output_far, "far")]:
                assert torch.all(torch.isfinite(output)), f"Output should be finite for {desc} points"

                constraint = manifold.dot(output, output) + 1.0 / manifold.c
                assert torch.allclose(constraint, torch.zeros_like(constraint), atol=1e-3), \
                    f"Output should satisfy constraint for {desc} points"

            results.add_pass("activation_adaptive_strategy")
        except Exception as e:
            results.add_fail("activation_adaptive_strategy", str(e))

        # 测试3：几何感知Dropout
        try:
            dropout = HyperbolicDropout(manifold, p=0.5)

            x = manifold.random_point(10, 8)
            dropout.train()  # 设置为训练模式

            output = dropout(x)

            # 检查输出有效性
            assert torch.all(torch.isfinite(output)), "Dropout output should be finite"

            constraint = manifold.dot(output, output) + 1.0 / manifold.c
            assert torch.allclose(constraint, torch.zeros_like(constraint), atol=1e-3), \
                "Dropout should preserve hyperboloid constraint"

            # 测试评估模式
            dropout.eval()
            output_eval = dropout(x)
            assert torch.allclose(x, output_eval), "Dropout should be identity in eval mode"

            results.add_pass("activation_geometric_dropout")
        except Exception as e:
            results.add_fail("activation_geometric_dropout", str(e))

    except ImportError as e:
        results.add_fail("activation_import", f"Failed to import activation modules: {e}")
    except Exception as e:
        results.add_fail("activation_general", f"Unexpected error: {e}")


def test_main_model_improvements(results: TestResults):
    """测试主模型的改进"""
    logger.info("Testing main model improvements...")

    try:
        from models.hgcn import HGCN, create_hgcn_from_config

        # 测试1：配置验证
        try:
            # 有效配置
            valid_config = {
                'model': {
                    'input_dim': 1280,
                    'hidden_dims': [256, 128],
                    'curvature': 1.0,
                    'learnable_curvature': True,
                    'aggregation_mode': 'mean',
                    'activation_type': 'relu',
                    'dropout': 0.1,
                    'decoder': {'type': 'distance', 'temperature': 'learnable'}
                }
            }

            model = create_hgcn_from_config(valid_config)
            assert model is not None, "Should create model with valid config"

            results.add_pass("model_config_validation_valid")
        except Exception as e:
            results.add_fail("model_config_validation_valid", str(e))

        # 测试2：无效配置的处理
        try:
            invalid_configs = [
                # 负的input_dim
                {'model': {'input_dim': -10}},
                # 空的hidden_dims
                {'model': {'hidden_dims': []}},
                # 无效的aggregation_mode
                {'model': {'aggregation_mode': 'invalid_mode'}},
                # 无效的dropout值
                {'model': {'dropout': 1.5}},
            ]

            for i, invalid_config in enumerate(invalid_configs):
                try:
                    model = create_hgcn_from_config(invalid_config)
                    results.add_fail("model_config_validation_invalid",
                                   f"Should reject invalid config {i}, but didn't")
                    return
                except ValueError:
                    # Expected behavior
                    pass

            results.add_pass("model_config_validation_invalid")
        except Exception as e:
            results.add_fail("model_config_validation_invalid", str(e))

        # 测试3：流形创建的边界情况
        try:
            # 测试每层曲率配置
            model = HGCN(
                input_dim=128,
                hidden_dims=[64, 32],
                curvature=[1.0, 2.0],
                curvature_per_layer=True
            )

            curvatures = model.get_curvatures()
            assert len(curvatures) == 2, f"Expected 2 layer curvatures, got {len(curvatures)}"

            results.add_pass("model_manifold_creation")
        except Exception as e:
            results.add_fail("model_manifold_creation", str(e))

        # 测试4：统计信息收集的鲁棒性
        try:
            model = HGCN(input_dim=64, hidden_dims=[32, 16])

            # 获取统计信息（即使没有运行前向传播）
            stats = model.get_statistics()

            required_keys = ['forward_count', 'edge_prediction_count', 'curvatures', 'model_info']
            for key in required_keys:
                assert key in stats, f"Statistics should contain {key}"

            # 测试内存使用情况
            memory_info = model.get_memory_usage()
            assert 'total_params' in memory_info, "Memory info should contain total_params"
            assert memory_info['total_params'] > 0, "Should have some parameters"

            results.add_pass("model_statistics_robustness")
        except Exception as e:
            results.add_fail("model_statistics_robustness", str(e))

        # 测试5：简单前向传播
        try:
            model = HGCN(
                input_dim=16,
                hidden_dims=[8, 4],
                curvature=1.0,
                learnable_curvature=False,
                dropout=0.0  # 关闭dropout简化测试
            )

            batch_size = 6
            num_edges = 10

            # 创建输入
            x = torch.randn(batch_size, 16)  # ESM-2特征
            edge_index = torch.randint(0, batch_size, (2, num_edges))

            # 前向传播
            embeddings = model(x, edge_index, return_embeddings=True)

            # 检查输出
            expected_shape = (batch_size, 4 + 1)  # 最后一层是4维
            assert embeddings.shape == expected_shape, f"Expected shape {expected_shape}, got {embeddings.shape}"

            assert torch.all(torch.isfinite(embeddings)), "Embeddings should be finite"

            # 测试链路预测
            test_edges = torch.tensor([[0, 1], [2, 3]]).t()
            scores = model.predict_links(embeddings, test_edges)

            assert scores.shape == (2,), f"Expected 2 scores, got shape {scores.shape}"
            assert torch.all(torch.isfinite(scores)), "Prediction scores should be finite"

            results.add_pass("model_forward_pass")
        except Exception as e:
            results.add_fail("model_forward_pass", str(e))

    except ImportError as e:
        results.add_fail("model_import", f"Failed to import main model modules: {e}")
    except Exception as e:
        results.add_fail("model_general", f"Unexpected error: {e}")


def run_all_tests() -> bool:
    """运行所有测试"""
    logger.info("Starting comprehensive test suite for HGCN fixes...")
    logger.info("="*60)

    results = TestResults()

    # 运行各个测试模块
    test_functions = [
        test_geometry_numerical_stability,
        test_aggregator_fallback_mechanism,
        test_linear_layer_dimension_handling,
        test_hgcn_layer_improvements,
        test_activation_improvements,
        test_main_model_improvements,
    ]

    for test_func in test_functions:
        try:
            test_func(results)
        except Exception as e:
            results.add_fail(test_func.__name__, f"Test function crashed: {e}")
            logger.error(f"Test function {test_func.__name__} crashed:")
            logger.error(traceback.format_exc())

    # 输出汇总
    success = results.summary()

    if success:
        logger.info("\n🎉 All tests passed! The fixes appear to be working correctly.")
    else:
        logger.error("\n💥 Some tests failed. Please check the errors above.")

    return success


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Test HGCN fixes')
    parser.add_argument('--test', choices=[
        'geometry', 'aggregator', 'linear', 'hgcn', 'activation', 'model', 'all'
    ], default='all', help='Which test to run')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    results = TestResults()

    # 根据选择运行特定测试
    if args.test == 'all':
        success = run_all_tests()
    else:
        test_map = {
            'geometry': test_geometry_numerical_stability,
            'aggregator': test_aggregator_fallback_mechanism,
            'linear': test_linear_layer_dimension_handling,
            'hgcn': test_hgcn_layer_improvements,
            'activation': test_activation_improvements,
            'model': test_main_model_improvements,
        }

        test_func = test_map[args.test]
        logger.info(f"Running specific test: {args.test}")

        try:
            test_func(results)
            success = results.summary()
        except Exception as e:
            results.add_fail(args.test, f"Test crashed: {e}")
            logger.error(f"Test {args.test} crashed:")
            logger.error(traceback.format_exc())
            success = False

    # 返回退出码
    return 0 if success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
