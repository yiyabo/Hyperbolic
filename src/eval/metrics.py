"""
评估指标模块

该模块实现了链路预测任务的各种评估指标，包括：
- AUPR (Area Under Precision-Recall Curve)
- AUROC (Area Under ROC Curve)
- Hits@K (排序质量评估)
- 校准曲线和可靠性图
- 平均精度和排序指标
- 统计显著性检验
- 不平衡数据处理
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


class LinkPredictionMetrics:
    """
    链路预测评估指标计算器

    提供全面的链路预测性能评估，包括：
    - 分类指标 (AUPR, AUROC, F1等)
    - 排序指标 (Hits@K, MRR等)
    - 校准指标 (可靠性, ECE等)
    - 统计检验
    """

    def __init__(
        self,
        k_values: List[int] = None,
        threshold_selection: str = "youden",  # youden, f1, precision_recall
        calibration_bins: int = 10,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95
    ):
        self.k_values = k_values or [10, 20, 50, 100]
        self.threshold_selection = threshold_selection
        self.calibration_bins = calibration_bins
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        return_curves: bool = False,
        compute_bootstrap: bool = False
    ) -> Dict[str, Any]:
        """
        计算所有评估指标

        Args:
            y_true: (N,) 真实标签 (0或1)
            y_scores: (N,) 预测分数
            return_curves: 是否返回ROC和PR曲线
            compute_bootstrap: 是否计算bootstrap置信区间

        Returns:
            包含所有指标的字典
        """
        results = {}

        # 基本分类指标
        results.update(self.compute_classification_metrics(y_true, y_scores))

        # 排序指标
        results.update(self.compute_ranking_metrics(y_true, y_scores))

        # 校准指标
        results.update(self.compute_calibration_metrics(y_true, y_scores))

        # 曲线数据
        if return_curves:
            results['curves'] = self.compute_curves(y_true, y_scores)

        # Bootstrap置信区间
        if compute_bootstrap:
            results['bootstrap_ci'] = self.compute_bootstrap_ci(y_true, y_scores)

        return results

    def compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, float]:
        """计算分类指标"""
        metrics = {}

        # AUPR (主要指标)
        try:
            metrics['aupr'] = average_precision_score(y_true, y_scores)
        except Exception as e:
            logger.warning(f"AUPR computation failed: {e}")
            metrics['aupr'] = 0.0

        # AUROC
        try:
            if len(np.unique(y_true)) > 1:
                metrics['auroc'] = roc_auc_score(y_true, y_scores)
            else:
                metrics['auroc'] = 0.5  # 单一类别情况
        except Exception as e:
            logger.warning(f"AUROC computation failed: {e}")
            metrics['auroc'] = 0.5

        # 最优阈值和对应指标
        optimal_threshold = self.find_optimal_threshold(y_true, y_scores)
        y_pred = (y_scores >= optimal_threshold).astype(int)

        metrics['optimal_threshold'] = optimal_threshold
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # 平衡准确率
        tn, fp, fn, tp = self._confusion_matrix_elements(y_true, y_pred)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['balanced_accuracy'] = (sensitivity + specificity) / 2
        metrics['sensitivity'] = sensitivity
        metrics['specificity'] = specificity

        return metrics

    def compute_ranking_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, float]:
        """计算排序指标"""
        metrics = {}

        # Hits@K
        for k in self.k_values:
            hits_k = self.hits_at_k(y_true, y_scores, k)
            metrics[f'hits@{k}'] = hits_k

        # Mean Reciprocal Rank (MRR)
        metrics['mrr'] = self.mean_reciprocal_rank(y_true, y_scores)

        # Mean Average Precision (MAP)
        metrics['map'] = self.mean_average_precision(y_true, y_scores)

        # NDCG@K
        for k in self.k_values:
            ndcg_k = self.ndcg_at_k(y_true, y_scores, k)
            metrics[f'ndcg@{k}'] = ndcg_k

        return metrics

    def compute_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, float]:
        """计算校准指标"""
        metrics = {}

        # 将分数转换为概率
        y_prob = self._scores_to_probabilities(y_scores)

        # 校准曲线
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=self.calibration_bins, strategy='uniform'
            )

            # Expected Calibration Error (ECE)
            ece = self._compute_ece(y_true, y_prob, self.calibration_bins)
            metrics['ece'] = ece

            # Maximum Calibration Error (MCE)
            mce = self._compute_mce(y_true, y_prob, self.calibration_bins)
            metrics['mce'] = mce

            # Reliability (校准斜率)
            if len(fraction_of_positives) > 1 and len(mean_predicted_value) > 1:
                slope, intercept, r_value, _, _ = stats.linregress(
                    mean_predicted_value, fraction_of_positives
                )
                metrics['calibration_slope'] = slope
                metrics['calibration_intercept'] = intercept
                metrics['calibration_r2'] = r_value ** 2
            else:
                metrics['calibration_slope'] = 1.0
                metrics['calibration_intercept'] = 0.0
                metrics['calibration_r2'] = 1.0

        except Exception as e:
            logger.warning(f"Calibration metrics computation failed: {e}")
            metrics['ece'] = 1.0
            metrics['mce'] = 1.0
            metrics['calibration_slope'] = 1.0
            metrics['calibration_intercept'] = 0.0
            metrics['calibration_r2'] = 0.0

        return metrics

    def compute_curves(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """计算ROC和PR曲线"""
        curves = {}

        # ROC曲线
        try:
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
            curves['roc'] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': roc_thresholds
            }
        except Exception as e:
            logger.warning(f"ROC curve computation failed: {e}")
            curves['roc'] = {'fpr': np.array([0, 1]), 'tpr': np.array([0, 1]), 'thresholds': np.array([1, 0])}

        # PR曲线
        try:
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
            curves['pr'] = {
                'precision': precision,
                'recall': recall,
                'thresholds': pr_thresholds
            }
        except Exception as e:
            logger.warning(f"PR curve computation failed: {e}")
            baseline = np.mean(y_true)
            curves['pr'] = {
                'precision': np.array([baseline, baseline]),
                'recall': np.array([1, 0]),
                'thresholds': np.array([0])
            }

        # 校准曲线
        try:
            y_prob = self._scores_to_probabilities(y_scores)
            fraction_pos, mean_pred = calibration_curve(
                y_true, y_prob, n_bins=self.calibration_bins
            )
            curves['calibration'] = {
                'fraction_of_positives': fraction_pos,
                'mean_predicted_value': mean_pred
            }
        except Exception as e:
            logger.warning(f"Calibration curve computation failed: {e}")
            curves['calibration'] = {
                'fraction_of_positives': np.linspace(0, 1, 10),
                'mean_predicted_value': np.linspace(0, 1, 10)
            }

        return curves

    def compute_bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """计算bootstrap置信区间"""
        n_samples = len(y_true)
        bootstrap_aupr = []
        bootstrap_auroc = []

        for _ in range(self.bootstrap_samples):
            # Bootstrap采样
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_scores_boot = y_scores[indices]

            # 计算指标
            try:
                aupr_boot = average_precision_score(y_true_boot, y_scores_boot)
                bootstrap_aupr.append(aupr_boot)
            except:
                pass

            try:
                if len(np.unique(y_true_boot)) > 1:
                    auroc_boot = roc_auc_score(y_true_boot, y_scores_boot)
                    bootstrap_auroc.append(auroc_boot)
            except:
                pass

        # 计算置信区间
        alpha = 1 - self.confidence_level
        ci = {}

        if bootstrap_aupr:
            aupr_lower = np.percentile(bootstrap_aupr, 100 * alpha / 2)
            aupr_upper = np.percentile(bootstrap_aupr, 100 * (1 - alpha / 2))
            ci['aupr'] = (aupr_lower, aupr_upper)

        if bootstrap_auroc:
            auroc_lower = np.percentile(bootstrap_auroc, 100 * alpha / 2)
            auroc_upper = np.percentile(bootstrap_auroc, 100 * (1 - alpha / 2))
            ci['auroc'] = (auroc_lower, auroc_upper)

        return ci

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> float:
        """寻找最优分类阈值"""
        if self.threshold_selection == "youden":
            # Youden's J statistic
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            return thresholds[optimal_idx]

        elif self.threshold_selection == "f1":
            # 最大F1分数
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        elif self.threshold_selection == "precision_recall":
            # 精确率=召回率的点
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            diff = np.abs(precision - recall)
            optimal_idx = np.argmin(diff)
            return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        else:
            return 0.5

    def hits_at_k(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        k: int
    ) -> float:
        """计算Hits@K"""
        # 按分数排序
        sorted_indices = np.argsort(y_scores)[::-1]

        # 取前K个
        top_k_indices = sorted_indices[:k]

        # 计算命中数
        hits = np.sum(y_true[top_k_indices])

        # 总的正样本数
        total_positives = np.sum(y_true)

        return hits / total_positives if total_positives > 0 else 0.0

    def mean_reciprocal_rank(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> float:
        """计算平均倒数排名 (MRR)"""
        # 按分数排序
        sorted_indices = np.argsort(y_scores)[::-1]
        ranks = np.argsort(sorted_indices) + 1  # 排名从1开始

        # 只考虑正样本的排名
        positive_ranks = ranks[y_true == 1]

        if len(positive_ranks) == 0:
            return 0.0

        # 计算倒数排名的平均值
        mrr = np.mean(1.0 / positive_ranks)
        return mrr

    def mean_average_precision(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> float:
        """计算平均精度均值 (MAP)"""
        # 按分数排序
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_labels = y_true[sorted_indices]

        # 计算每个位置的精度
        precisions = []
        num_positives = 0

        for i, label in enumerate(sorted_labels):
            if label == 1:
                num_positives += 1
                precision = num_positives / (i + 1)
                precisions.append(precision)

        return np.mean(precisions) if precisions else 0.0

    def ndcg_at_k(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        k: int
    ) -> float:
        """计算NDCG@K"""
        # 按分数排序
        sorted_indices = np.argsort(y_scores)[::-1][:k]

        # DCG@K
        dcg = 0
        for i, idx in enumerate(sorted_indices):
            if y_true[idx] == 1:
                dcg += 1 / np.log2(i + 2)  # i从0开始，所以+2

        # IDCG@K (理想情况下的DCG)
        num_positives = min(k, np.sum(y_true))
        idcg = sum(1 / np.log2(i + 2) for i in range(num_positives))

        return dcg / idcg if idcg > 0 else 0.0

    def _scores_to_probabilities(self, scores: np.ndarray) -> np.ndarray:
        """将分数转换为概率"""
        # 使用sigmoid函数
        return 1 / (1 + np.exp(-scores))

    def _compute_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
        """计算Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _compute_mce(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
        """计算Maximum Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        max_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)

            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                max_error = max(max_error, np.abs(avg_confidence_in_bin - accuracy_in_bin))

        return max_error

    def _confusion_matrix_elements(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
        """计算混淆矩阵元素"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tn, fp, fn, tp


class BatchMetricsComputer:
    """批量指标计算器，用于大规模评估"""

    def __init__(self, metrics_computer: LinkPredictionMetrics):
        self.metrics_computer = metrics_computer

    def compute_batch_metrics(
        self,
        y_true_batches: List[np.ndarray],
        y_scores_batches: List[np.ndarray],
        aggregate_method: str = "weighted"  # weighted, macro, micro
    ) -> Dict[str, float]:
        """
        批量计算指标

        Args:
            y_true_batches: 真实标签批次列表
            y_scores_batches: 预测分数批次列表
            aggregate_method: 聚合方法

        Returns:
            聚合后的指标字典
        """
        batch_metrics = []
        batch_sizes = []

        # 计算每个批次的指标
        for y_true_batch, y_scores_batch in zip(y_true_batches, y_scores_batches):
            metrics = self.metrics_computer.compute_classification_metrics(
                y_true_batch, y_scores_batch
            )
            batch_metrics.append(metrics)
            batch_sizes.append(len(y_true_batch))

        # 聚合指标
        if aggregate_method == "weighted":
            return self._weighted_average(batch_metrics, batch_sizes)
        elif aggregate_method == "macro":
            return self._macro_average(batch_metrics)
        elif aggregate_method == "micro":
            # 微平均：先合并所有数据再计算
            all_y_true = np.concatenate(y_true_batches)
            all_y_scores = np.concatenate(y_scores_batches)
            return self.metrics_computer.compute_classification_metrics(all_y_true, all_y_scores)
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate_method}")

    def _weighted_average(self, batch_metrics: List[Dict], batch_sizes: List[int]) -> Dict[str, float]:
        """加权平均"""
        total_size = sum(batch_sizes)
        aggregated = {}

        for key in batch_metrics[0].keys():
            weighted_sum = sum(
                metrics[key] * size for metrics, size in zip(batch_metrics, batch_sizes)
            )
            aggregated[key] = weighted_sum / total_size

        return aggregated

    def _macro_average(self, batch_metrics: List[Dict]) -> Dict[str, float]:
        """宏平均"""
        aggregated = {}

        for key in batch_metrics[0].keys():
            values = [metrics[key] for metrics in batch_metrics]
            aggregated[key] = np.mean(values)

        return aggregated


def compute_significance_test(
    y_true: np.ndarray,
    y_scores_1: np.ndarray,
    y_scores_2: np.ndarray,
    metric: str = "aupr",
    n_bootstrap: int = 1000
) -> Dict[str, float]:
    """
    比较两个模型性能的统计显著性检验

    Args:
        y_true: 真实标签
        y_scores_1: 模型1的预测分数
        y_scores_2: 模型2的预测分数
        metric: 比较的指标名称
        n_bootstrap: bootstrap采样次数

    Returns:
        包含检验结果的字典
    """
    metrics_computer = LinkPredictionMetrics()

    # 计算原始指标差异
    if metric == "aupr":
        score_1 = average_precision_score(y_true, y_scores_1)
        score_2 = average_precision_score(y_true, y_scores_2)
    elif metric == "auroc":
        score_1 = roc_auc_score(y_true, y_scores_1) if len(np.unique(y_true)) > 1 else 0.5
        score_2 = roc_auc_score(y_true, y_scores_2) if len(np.unique(y_true)) > 1 else 0.5
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    observed_diff = score_1 - score_2

    # Bootstrap采样
    n_samples = len(y_true)
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_scores_1_boot = y_scores_1[indices]
        y_scores_2_boot = y_scores_2[indices]

        try:
            if metric == "aupr":
                score_1_boot = average_precision_score(y_true_boot, y_scores_1_boot)
                score_2_boot = average_precision_score(y_true_boot, y_scores_2_boot)
            else:  # auroc
                if len(np.unique(y_true_boot)) > 1:
                    score_1_boot = roc_auc_score(y_true_boot, y_scores_1_boot)
                    score_2_boot = roc_auc_score(y_true_boot, y_scores_2_boot)
                else:
                    continue

            bootstrap_diffs.append(score_1_boot - score_2_boot)
        except:
            continue

    bootstrap_diffs = np.array(bootstrap_diffs)

    # 计算p值 (双侧检验)
    if len(bootstrap_diffs) > 0:
        p_value = 2 * min(
            np.mean(bootstrap_diffs <= 0),
            np.mean(bootstrap_diffs >= 0)
        )
    else:
        p_value = 1.0

    return {
        f'{metric}_1': score_1,
        f'{metric}_2': score_2,
        'difference': observed_diff,
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'n_bootstrap_samples': len(bootstrap_diffs)
    }


# 便捷函数
def evaluate_link_prediction(
    y_true: Union[np.ndarray, torch.Tensor],
    y_scores: Union[np.ndarray, torch.Tensor],
    **kwargs
) -> Dict[str, Any]:
    """
    链路预测评估的便捷函数

    Args:
        y_true: 真实标签
        y_scores: 预测分数
        **kwargs: LinkPredictionMetrics的参数

    Returns:
        评估结果字典
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()

    metrics_computer = LinkPredictionMetrics(**kwargs)
    return metrics_computer.compute_all_metrics(y_true, y_scores)


def quick_evaluate(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """快速评估（只计算核心指标）"""
    results = {}

    try:
        results['aupr'] = average_precision_score(y_true, y_scores)
    except:
        results['aupr'] = 0.0

    try:
        if len(np.unique(y_true)) > 1:
            results['auroc'] = roc_auc_score(y_true, y_scores)
        else:
            results['auroc'] = 0.5
    except:
        results['auroc'] = 0.5

    # 简单hits@10
    if len(y_scores) >= 10:
        sorted_indices = np.argsort(y_scores)[::-1][:10]
        results['hits@10'] = np.sum(y_true[sorted_indices]) / max(np.sum(y_true), 1)
    else:
        results['hits@10'] = 0.0

    return results
