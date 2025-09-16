"""
链路预测评估流程模块

该模块实现了完整的HGCN链路预测评估流程，包括：
- 模型加载和推理
- 多数据集评估
- 全面的指标计算
- 跨物种/跨数据集验证
- 消融实验支持
- 结果可视化和报告生成
- 统计显著性分析

评估流程：
1. 加载训练好的模型
2. 加载测试数据和特征
3. 生成节点嵌入
4. 计算所有节点对的相似性分数
5. 与真实标签对比计算指标
6. 生成详细报告和可视化
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import logging

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import yaml
from tqdm import tqdm

# 项目模块
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.hgcn import HGCN, create_hgcn_from_config
from src.eval.metrics import LinkPredictionMetrics, evaluate_link_prediction, compute_significance_test
from src.dataio.neg_sampling import create_negative_sampler
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """评估配置"""
    # 模型和数据路径
    model_checkpoint: str = "checkpoints/best.pt"
    test_data_paths: List[str] = None
    feature_dirs: List[str] = None

    # 评估设置
    batch_size: int = 10000
    k_values: List[int] = None
    threshold_selection: str = "youden"

    # 负采样设置
    negative_sampling: Dict = None
    neg_pos_ratio: int = 1

    # 输出设置
    output_dir: str = "evaluation_results"
    save_predictions: bool = True
    save_embeddings: bool = False
    save_visualizations: bool = True

    # 统计分析
    compute_bootstrap: bool = True
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95

    # 设备和性能
    device: str = "auto"
    chunk_size: int = 1000
    num_workers: int = 4

    # 其他
    seed: int = 42
    debug: bool = False

    def __post_init__(self):
        if self.test_data_paths is None:
            self.test_data_paths = []
        if self.feature_dirs is None:
            self.feature_dirs = []
        if self.k_values is None:
            self.k_values = [10, 20, 50, 100]
        if self.negative_sampling is None:
            self.negative_sampling = {"strategy": "uniform", "ratio": 1}


class LinkPredictionEvaluator:
    """链路预测评估器"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = self._setup_device()

        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)

        # 指标计算器
        self.metrics_computer = LinkPredictionMetrics(
            k_values=self.config.k_values,
            threshold_selection=self.config.threshold_selection,
            bootstrap_samples=self.config.bootstrap_samples,
            confidence_level=self.config.confidence_level
        )

        # 存储结果
        self.results = {}
        self.embeddings = {}
        self.predictions = {}

        logger.info(f"Evaluator initialized with device: {self.device}")

    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)

        logger.info(f"Using device: {device}")
        return device

    def load_model(self, checkpoint_path: str) -> HGCN:
        """加载训练好的模型"""
        logger.info(f"Loading model from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 从检查点中重建模型配置
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            model = create_hgcn_from_config({'model': model_config.get('model', {})})
        else:
            # 如果没有配置，使用默认配置
            logger.warning("No config found in checkpoint, using default model config")
            model = create_hgcn_from_config({})

        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        logger.info(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
        return model

    def load_test_data(self, data_path: str, feature_path: str = None) -> Tuple[Dict, torch.Tensor]:
        """加载测试数据"""
        logger.info(f"Loading test data from {data_path}")

        # 加载图数据
        graph_data = torch.load(data_path, map_location=self.device)

        # 加载特征
        if feature_path and os.path.exists(feature_path):
            features = torch.load(feature_path, map_location=self.device)
        else:
            # 生成随机特征（用于测试）
            logger.warning(f"Feature file not found: {feature_path}, using random features")
            features = torch.randn(graph_data.num_nodes, 1280, device=self.device)  # ESM-2 650M

        logger.info(f"Loaded graph with {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
        return graph_data, features

    def generate_embeddings(self, model: HGCN, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """生成节点嵌入"""
        logger.info("Generating node embeddings...")

        with torch.no_grad():
            embeddings = model(features, edge_index)

        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def evaluate_single_dataset(
        self,
        model: HGCN,
        graph_data: Dict,
        features: torch.Tensor,
        dataset_name: str
    ) -> Dict[str, Any]:
        """评估单个数据集"""
        logger.info(f"Evaluating dataset: {dataset_name}")

        start_time = time.time()
        results = {'dataset_name': dataset_name}

        try:
            # 生成节点嵌入
            edge_index = graph_data.edge_index
            embeddings = self.generate_embeddings(model, features, edge_index)

            # 存储嵌入（如果需要）
            if self.config.save_embeddings:
                self.embeddings[dataset_name] = embeddings.cpu()

            # 准备评估边和标签
            eval_results = self._evaluate_link_prediction(
                model, embeddings, graph_data, dataset_name
            )
            results.update(eval_results)

            # 计算额外统计信息
            results['evaluation_time'] = time.time() - start_time
            results['num_nodes'] = embeddings.size(0)
            results['num_edges'] = edge_index.size(1)
            results['embedding_dim'] = embeddings.size(1)

            # 模型特定统计
            model_stats = model.get_statistics()
            results['model_stats'] = model_stats

        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results

        results['success'] = True
        logger.info(f"Completed evaluation of {dataset_name} in {results['evaluation_time']:.2f}s")

        return results

    def _evaluate_link_prediction(
        self,
        model: HGCN,
        embeddings: torch.Tensor,
        graph_data: Dict,
        dataset_name: str
    ) -> Dict[str, Any]:
        """执行链路预测评估"""

        # 获取正边（真实连接）
        pos_edges = graph_data.edge_index
        num_pos = pos_edges.size(1)

        # 生成负边
        neg_edges = self._generate_negative_edges(
            embeddings.size(0), pos_edges, num_pos * self.config.neg_pos_ratio
        )

        # 计算分数
        logger.info("Computing link prediction scores...")
        pos_scores = self._compute_scores_batched(model, embeddings, pos_edges)
        neg_scores = self._compute_scores_batched(model, embeddings, neg_edges)

        # 组合标签和分数
        y_true = np.concatenate([
            np.ones(len(pos_scores)),
            np.zeros(len(neg_scores))
        ])
        y_scores = np.concatenate([pos_scores, neg_scores])

        # 存储预测（如果需要）
        if self.config.save_predictions:
            self.predictions[dataset_name] = {
                'y_true': y_true,
                'y_scores': y_scores,
                'pos_edges': pos_edges.cpu().numpy(),
                'neg_edges': neg_edges.cpu().numpy()
            }

        # 计算所有指标
        logger.info("Computing evaluation metrics...")
        metrics = self.metrics_computer.compute_all_metrics(
            y_true, y_scores,
            return_curves=True,
            compute_bootstrap=self.config.compute_bootstrap
        )

        return metrics

    def _compute_scores_batched(
        self,
        model: HGCN,
        embeddings: torch.Tensor,
        edges: torch.Tensor
    ) -> np.ndarray:
        """批量计算边分数"""
        scores = []
        num_edges = edges.size(1)

        with torch.no_grad():
            for i in range(0, num_edges, self.config.batch_size):
                end_i = min(i + self.config.batch_size, num_edges)
                batch_edges = edges[:, i:end_i]

                batch_scores = model.predict_links(embeddings, batch_edges)
                scores.append(batch_scores.cpu().numpy())

        return np.concatenate(scores)

    def _generate_negative_edges(
        self,
        num_nodes: int,
        pos_edges: torch.Tensor,
        num_neg: int
    ) -> torch.Tensor:
        """生成负边"""
        logger.info(f"Generating {num_neg} negative edges...")

        # 创建正边集合（用于避免重复）
        pos_edge_set = set()
        pos_edges_np = pos_edges.cpu().numpy()
        for i in range(pos_edges_np.shape[1]):
            u, v = pos_edges_np[:, i]
            pos_edge_set.add((min(u, v), max(u, v)))

        neg_edges = []
        max_attempts = num_neg * 10
        attempts = 0

        while len(neg_edges) < num_neg and attempts < max_attempts:
            # 随机采样边
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)

            if u != v:  # 避免自环
                edge = (min(u, v), max(u, v))
                if edge not in pos_edge_set:
                    neg_edges.append([u, v])
                    pos_edge_set.add(edge)  # 避免重复采样

            attempts += 1

        if len(neg_edges) < num_neg:
            logger.warning(f"Only generated {len(neg_edges)} negative edges out of {num_neg} requested")

        neg_edges_tensor = torch.tensor(neg_edges, device=self.device, dtype=torch.long).t()
        return neg_edges_tensor

    def evaluate_all_datasets(self, model: HGCN) -> Dict[str, Any]:
        """评估所有数据集"""
        logger.info("Starting evaluation of all datasets...")

        all_results = {}
        dataset_names = []

        # 评估每个数据集
        for i, (data_path, feature_path) in enumerate(zip(
            self.config.test_data_paths,
            self.config.feature_dirs
        )):
            dataset_name = f"dataset_{i}_{Path(data_path).stem}"
            dataset_names.append(dataset_name)

            try:
                graph_data, features = self.load_test_data(data_path, feature_path)
                results = self.evaluate_single_dataset(model, graph_data, features, dataset_name)
                all_results[dataset_name] = results

            except Exception as e:
                logger.error(f"Failed to evaluate {data_path}: {e}")
                all_results[dataset_name] = {
                    'error': str(e),
                    'success': False,
                    'dataset_name': dataset_name
                }

        # 计算跨数据集统计
        if len([r for r in all_results.values() if r.get('success', False)]) > 1:
            all_results['cross_dataset_stats'] = self._compute_cross_dataset_stats(all_results)

        self.results = all_results
        return all_results

    def _compute_cross_dataset_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算跨数据集统计"""
        logger.info("Computing cross-dataset statistics...")

        successful_results = [r for r in results.values() if r.get('success', False)]

        if len(successful_results) < 2:
            return {}

        # 收集主要指标
        metrics_to_aggregate = ['aupr', 'auroc', 'f1', 'accuracy']
        aggregated = {}

        for metric in metrics_to_aggregate:
            values = [r.get(metric, 0) for r in successful_results if metric in r]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)

        # 数据集统计
        aggregated['num_datasets'] = len(successful_results)
        aggregated['total_nodes'] = sum(r.get('num_nodes', 0) for r in successful_results)
        aggregated['total_edges'] = sum(r.get('num_edges', 0) for r in successful_results)

        return aggregated

    def compare_models(
        self,
        model_checkpoints: List[str],
        model_names: List[str] = None
    ) -> Dict[str, Any]:
        """比较多个模型的性能"""
        logger.info(f"Comparing {len(model_checkpoints)} models...")

        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(model_checkpoints))]

        comparison_results = {}
        all_predictions = {}

        # 评估每个模型
        for checkpoint_path, model_name in zip(model_checkpoints, model_names):
            logger.info(f"Evaluating {model_name}...")

            model = self.load_model(checkpoint_path)
            results = self.evaluate_all_datasets(model)

            comparison_results[model_name] = results
            all_predictions[model_name] = self.predictions.copy()

        # 统计显著性检验
        if len(model_checkpoints) == 2 and len(self.config.test_data_paths) > 0:
            significance_tests = self._compute_model_significance_tests(
                all_predictions, model_names
            )
            comparison_results['significance_tests'] = significance_tests

        return comparison_results

    def _compute_model_significance_tests(
        self,
        all_predictions: Dict[str, Dict],
        model_names: List[str]
    ) -> Dict[str, Any]:
        """计算模型间的统计显著性"""
        if len(model_names) != 2:
            return {}

        model_1, model_2 = model_names
        tests = {}

        # 对每个数据集进行显著性检验
        for dataset_name in all_predictions[model_1].keys():
            if dataset_name in all_predictions[model_2]:
                pred_1 = all_predictions[model_1][dataset_name]
                pred_2 = all_predictions[model_2][dataset_name]

                # AUPR显著性检验
                aupr_test = compute_significance_test(
                    pred_1['y_true'], pred_1['y_scores'], pred_2['y_scores'],
                    metric='aupr'
                )

                # AUROC显著性检验
                auroc_test = compute_significance_test(
                    pred_1['y_true'], pred_1['y_scores'], pred_2['y_scores'],
                    metric='auroc'
                )

                tests[dataset_name] = {
                    'aupr_test': aupr_test,
                    'auroc_test': auroc_test
                }

        return tests

    def generate_visualizations(self) -> Dict[str, str]:
        """生成可视化图表"""
        if not self.config.save_visualizations:
            return {}

        logger.info("Generating visualizations...")

        vis_paths = {}

        # 性能对比图
        if self.results:
            vis_paths.update(self._plot_performance_comparison())

        # 嵌入可视化
        if self.embeddings:
            vis_paths.update(self._plot_embeddings())

        # ROC和PR曲线
        if self.predictions:
            vis_paths.update(self._plot_roc_pr_curves())

        return vis_paths

    def _plot_performance_comparison(self) -> Dict[str, str]:
        """绘制性能对比图"""
        vis_paths = {}

        try:
            successful_results = [
                r for r in self.results.values()
                if isinstance(r, dict) and r.get('success', False)
            ]

            if len(successful_results) < 1:
                return {}

            # 提取指标
            metrics = ['aupr', 'auroc', 'f1', 'accuracy']
            dataset_names = [r['dataset_name'] for r in successful_results]

            # 创建性能对比图
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()

            for i, metric in enumerate(metrics):
                values = [r.get(metric, 0) for r in successful_results]

                axes[i].bar(range(len(dataset_names)), values)
                axes[i].set_title(f'{metric.upper()} Comparison')
                axes[i].set_xlabel('Dataset')
                axes[i].set_ylabel(metric.upper())
                axes[i].set_xticks(range(len(dataset_names)))
                axes[i].set_xticklabels(dataset_names, rotation=45)

                # 添加数值标签
                for j, v in enumerate(values):
                    axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

            plt.tight_layout()

            performance_path = os.path.join(self.config.output_dir, 'performance_comparison.png')
            plt.savefig(performance_path, dpi=300, bbox_inches='tight')
            plt.close()

            vis_paths['performance_comparison'] = performance_path

        except Exception as e:
            logger.error(f"Error plotting performance comparison: {e}")

        return vis_paths

    def _plot_embeddings(self) -> Dict[str, str]:
        """绘制嵌入可视化"""
        vis_paths = {}

        try:
            for dataset_name, embeddings in self.embeddings.items():
                # 转换为numpy
                emb_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings

                # t-SNE降维
                if emb_np.shape[0] > 1000:
                    # 大数据集先用PCA预处理
                    pca = PCA(n_components=50)
                    emb_reduced = pca.fit_transform(emb_np)
                else:
                    emb_reduced = emb_np

                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, emb_reduced.shape[0]-1))
                emb_2d = tsne.fit_transform(emb_reduced)

                # 绘制散点图
                plt.figure(figsize=(10, 8))
                plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6, s=20)
                plt.title(f'Embedding Visualization - {dataset_name}')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')

                embedding_path = os.path.join(self.config.output_dir, f'embedding_{dataset_name}.png')
                plt.savefig(embedding_path, dpi=300, bbox_inches='tight')
                plt.close()

                vis_paths[f'embedding_{dataset_name}'] = embedding_path

        except Exception as e:
            logger.error(f"Error plotting embeddings: {e}")

        return vis_paths

    def _plot_roc_pr_curves(self) -> Dict[str, str]:
        """绘制ROC和PR曲线"""
        vis_paths = {}

        try:
            for dataset_name, pred_data in self.predictions.items():
                y_true = pred_data['y_true']
                y_scores = pred_data['y_scores']

                # 计算曲线
                curves = self.metrics_computer.compute_curves(y_true, y_scores)

                # 绘制ROC和PR曲线
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # ROC曲线
                roc = curves['roc']
                ax1.plot(roc['fpr'], roc['tpr'], linewidth=2)
                ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                ax1.set_xlabel('False Positive Rate')
                ax1.set_ylabel('True Positive Rate')
                ax1.set_title('ROC Curve')
                ax1.grid(True, alpha=0.3)

                # PR曲线
                pr = curves['pr']
                ax2.plot(pr['recall'], pr['precision'], linewidth=2)
                ax2.axhline(y=np.mean(y_true), color='k', linestyle='--', alpha=0.5)
                ax2.set_xlabel('Recall')
                ax2.set_ylabel('Precision')
                ax2.set_title('Precision-Recall Curve')
                ax2.grid(True, alpha=0.3)

                plt.suptitle(f'ROC and PR Curves - {dataset_name}')
                plt.tight_layout()

                curves_path = os.path.join(self.config.output_dir, f'curves_{dataset_name}.png')
                plt.savefig(curves_path, dpi=300, bbox_inches='tight')
                plt.close()

                vis_paths[f'curves_{dataset_name}'] = curves_path

        except Exception as e:
            logger.error(f"Error plotting ROC/PR curves: {e}")

        return vis_paths

    def save_results(self, filename: str = "evaluation_results.json"):
        """保存评估结果"""
        results_path = os.path.join(self.config.output_dir, filename)

        # 准备可序列化的结果
        serializable_results = self._make_serializable(self.results)

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_path}")

        # 保存预测数据
        if self.config.save_predictions and self.predictions:
            pred_path = os.path.join(self.config.output_dir, "predictions.npz")
            np.savez_compressed(pred_path, **self.predictions)
            logger.info(f"Predictions saved to {pred_path}")

        # 保存嵌入数据
        if self.config.save_embeddings and self.embeddings:
            emb_path = os.path.join(self.config.output_dir, "embeddings.pt")
            torch.save(self.embeddings, emb_path)
            logger.info(f"Embeddings saved to {emb_path}")

    def _make_serializable(self, obj):
        """使对象可序列化"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    def generate_report(self) -> str:
        """生成评估报告"""
        report_path = os.path.join(self.config.output_dir, "evaluation_report.md")

        with open(report_path, 'w') as f:
            f.write("# HGCN Link Prediction Evaluation Report\n\n")
            f.write(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 配置信息
            f.write("## Configuration\n\n")
            f.write("```yaml\n")
            f.write(yaml.dump(asdict(self.config), default_flow_style=False))
            f.write("```\n\n")

            # 结果摘要
            if self.results:
                f.write("## Results Summary\n\n")
                successful_results = [
                    r for r in self.results.values()
                    if isinstance(r, dict) and r.get('success', False)
                ]

                if successful_results:
                    # 创建结果表格
                    f.write("| Dataset | AUPR | AUROC | F1 | Accuracy | Nodes | Edges |\n")
                    f.write("|---------|------|-------|----|---------:|------:|------:|\n")

                    for result in successful_results:
                        f.write(f"| {result.get('dataset_name', 'Unknown')} |")
                        f.write(f" {result.get('aupr', 0):.4f} |")
                        f.write(f" {result.get('auroc', 0):.4f} |")
                        f.write(f" {result.get('f1', 0):.4f} |")
                        f.write(f" {result.get('accuracy', 0):.4f} |")
                        f.write(f" {result.get('num_nodes', 0)} |")
                        f.write(f" {result.get('num_edges', 0)} |\n")

                    f.write("\n")

                    # 跨数据集统计
                    if 'cross_dataset_stats' in self.results:
                        stats = self.results['cross_dataset_stats']
                        f.write("## Cross-Dataset Statistics\n\n")
                        for key, value in stats.items():
                            if isinstance(value, float):
                                f.write(f"- **{key}**: {value:.4f}\n")
                            else:
                                f.write(f"- **{key}**: {value}\n")
                        f.write("\n")

            f.write("## Files Generated\n\n")
            f.write("- `evaluation_results.json`: Detailed results in JSON format\n")
            if self.config.save_predictions:
                f.write("- `predictions.npz`: Raw predictions for all datasets\n")
            if self.config.save_embeddings:
                f.write("- `embeddings.pt`: Node embeddings\n")
            if self.config.save_visualizations:
                f.write("- Various `.png` files: Performance plots and visualizations\n")

        logger.info(f"Report saved to {report_path}")
        return report_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HGCN Link Prediction Evaluation")
    parser.add_argument("--config", "-c", type=str, required=True,
                       help="Evaluation config file")
    parser.add_argument("--checkpoint", "-m", type=str, required=True,
                       help="Model checkpoint path")
    parser.add_argument("--output-dir", "-o", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--test-data", "-d", type=str, nargs="+", required=True,
                       help="Test data paths")
    parser.add_argument("--features", "-f", type=str, nargs="+",
                       help="Feature data paths")
    parser.add_argument("--batch-size", "-b", type=int, default=10000,
                       help="Batch size for evaluation")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple models")
    parser.add_argument("--no-bootstrap", action="store_true",
                       help="Skip bootstrap confidence intervals")

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置日志
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # 创建评估配置
        eval_config = EvaluationConfig(
            output_dir=args.output_dir,
            test_data_paths=args.test_data,
            feature_dirs=args.features or [None] * len(args.test_data),
            batch_size=args.batch_size,
            compute_bootstrap=not args.no_bootstrap,
            debug=args.debug,
            seed=args.seed
        )

        # 创建评估器
        evaluator = LinkPredictionEvaluator(eval_config)

        if args.compare and len(args.checkpoint.split(',')) > 1:
            # 比较多个模型
            checkpoints = args.checkpoint.split(',')
            model_names = [f"Model_{i+1}" for i in range(len(checkpoints))]

            logger.info(f"Comparing {len(checkpoints)} models...")
            results = evaluator.compare_models(checkpoints, model_names)
        else:
            # 评估单个模型
            logger.info("Evaluating single model...")
            model = evaluator.load_model(args.checkpoint)
            results = evaluator.evaluate_all_datasets(model)

        # 生成可视化
        vis_paths = evaluator.generate_visualizations()
        if vis_paths:
            logger.info(f"Generated {len(vis_paths)} visualizations")

        # 保存结果
        evaluator.save_results()

        # 生成报告
        report_path = evaluator.generate_report()

        # 打印摘要
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {eval_config.output_dir}")
        logger.info(f"Report generated: {report_path}")

        # 打印关键指标摘要
        if isinstance(results, dict) and not args.compare:
            successful_results = [
                r for r in results.values()
                if isinstance(r, dict) and r.get('success', False)
            ]
            if successful_results:
                logger.info("Performance Summary:")
                for result in successful_results:
                    name = result.get('dataset_name', 'Unknown')
                    aupr = result.get('aupr', 0)
                    auroc = result.get('auroc', 0)
                    logger.info(f"  {name}: AUPR={aupr:.4f}, AUROC={auroc:.4f}")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
