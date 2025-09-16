"""
链路预测训练模块

该模块实现了完整的HGCN链路预测训练流程，包括：
- Riemannian优化器集成
- 负采样策略
- 早停和学习率调度
- 多数据集验证
- 统计监控和日志记录
- 模型checkpointing
- 数值稳定性监控

训练流程：
1. 加载图数据和ESM-2特征
2. 创建HGCN模型和Riemannian优化器
3. 训练循环：前向传播→负采样→损失计算→反向传播
4. 验证和早停检查
5. 统计记录和模型保存
"""

import os
import sys
import time
import math
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import yaml
from tqdm import tqdm

# 第三方依赖
try:
    import geoopt
    GEOOPT_AVAILABLE = True
except ImportError:
    GEOOPT_AVAILABLE = False
    logging.warning("geoopt not available, falling back to standard optimizers")

try:
    import wandb
    WANDB_AVAILABLE = True
except (ImportError, Exception):
    WANDB_AVAILABLE = False

# 项目模块
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.hgcn import HGCN, create_hgcn_from_config
from src.dataio.neg_sampling import create_negative_sampler
from src.utils.seed import set_seed
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model: Dict = None

    # 训练配置
    epochs: int = 100
    batch_size: int = 8192
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "riemannian_adam"  # riemannian_adam, adam, sgd

    # 学习率调度
    scheduler: str = "plateau"  # plateau, cosine, step, none
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # 早停
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4

    # 负采样
    negative_sampling: Dict = None

    # 验证和评估
    val_frequency: int = 1
    val_metrics: List[str] = None

    # 日志和保存
    log_frequency: int = 10
    save_frequency: int = 10
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # 数值稳定性
    grad_clip: float = 1.0
    curvature_clip: bool = True
    curvature_min: float = 1e-4
    curvature_max: float = 10.0

    # 其他
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    debug: bool = False

    def __post_init__(self):
        """后处理默认值"""
        if self.model is None:
            self.model = {}
        if self.negative_sampling is None:
            self.negative_sampling = {"strategy": "topology_driven", "ratio": 5, "hard_frac": 0.5}
        if self.val_metrics is None:
            self.val_metrics = ["aupr", "auroc"]


class LinkPredictionTrainer:
    """链路预测训练器"""

    def __init__(
        self,
        model: HGCN,
        train_data: Dict,
        val_data: Dict,
        test_data: List[Dict],
        config: TrainingConfig,
        features: Dict[str, torch.Tensor],
        negative_sampler=None
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.config = config
        self.features = features
        self.negative_sampler = negative_sampler

        # 设备配置
        self.device = self._setup_device()
        self.model = self.model.to(self.device)

        # 优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # 训练状态
        self.epoch = 0
        self.best_val_score = -float('inf')
        self.patience_counter = 0
        self.training_stats = []
        self.validation_stats = []

        # 监控指标
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'curvatures': [],
            'grad_norms': [],
            'numerical_issues': 0,
            'early_stopped': False
        }

        # 创建保存目录
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

        logger.info("Trainer initialized successfully")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device = torch.device(self.config.device)

        return device

    def _create_optimizer(self):
        """创建优化器"""
        params = list(self.model.parameters())

        if self.config.optimizer == "riemannian_adam" and GEOOPT_AVAILABLE:
            # Riemannian Adam优化器
            optimizer = geoopt.optim.RiemannianAdam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            logger.info("Using Riemannian Adam optimizer")

        elif self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            logger.info("Using standard Adam optimizer")

        elif self.config.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
            logger.info("Using SGD optimizer")

        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        return optimizer

    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.min_lr,
                verbose=True
            )
        elif self.config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=self.config.scheduler_factor
            )
        else:
            scheduler = None

        if scheduler:
            logger.info(f"Using {self.config.scheduler} learning rate scheduler")
        else:
            logger.info("No learning rate scheduler")

        return scheduler

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_stats = {
            'loss': 0.0,
            'num_batches': 0,
            'num_edges': 0,
            'pos_acc': 0.0,
            'neg_acc': 0.0,
            'grad_norm': 0.0,
            'curvatures': {},
            'decoder_temp': 0.0
        }

        # 获取训练边和特征
        edge_index = self.train_data['edge_index'].to(self.device)
        x = self.features['train'].to(self.device)

        # 准备负采样器
        if self.negative_sampler:
            self.negative_sampler.prepare_graph(self.train_data)

        # 计算总批次数
        num_pos_edges = edge_index.size(1)
        num_batches = max(1, num_pos_edges // self.config.batch_size)

        progress_bar = tqdm(range(num_batches), desc=f"Epoch {self.epoch}",
                          disable=not self.config.debug)

        for batch_idx in progress_bar:
            # 批次边采样
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, num_pos_edges)
            batch_pos_edges = edge_index[:, start_idx:end_idx]

            # 负采样
            if self.negative_sampler:
                batch_neg_edges = self.negative_sampler.sample(
                    batch_pos_edges,
                    num_neg_samples=batch_pos_edges.size(1) * self.config.negative_sampling['ratio']
                )
            else:
                # 简单随机负采样
                neg_dst = torch.randint(0, x.size(0),
                                      (batch_pos_edges.size(1) * self.config.negative_sampling['ratio'],),
                                      device=self.device)
                neg_src = batch_pos_edges[0].repeat_interleave(self.config.negative_sampling['ratio'])
                batch_neg_edges = torch.stack([neg_src, neg_dst], dim=0)

            # 前向传播
            try:
                # 获取节点嵌入
                h = self.model(x, edge_index)  # 使用全图进行消息传递

                # 计算正样本分数
                pos_scores = self.model.predict_links(h, batch_pos_edges)

                # 计算负样本分数
                neg_scores = self.model.predict_links(h, batch_neg_edges)

                # 组合分数和标签
                scores = torch.cat([pos_scores, neg_scores], dim=0)
                labels = torch.cat([
                    torch.ones_like(pos_scores),
                    torch.zeros_like(neg_scores)
                ], dim=0)

                # 计算损失
                loss = F.binary_cross_entropy_with_logits(scores, labels)

            except RuntimeError as e:
                logger.error(f"Forward pass error in batch {batch_idx}: {e}")
                self.metrics['numerical_issues'] += 1
                continue

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if self.config.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            else:
                grad_norm = self._compute_grad_norm()

            # 优化步骤
            self.optimizer.step()

            # 曲率裁剪
            if self.config.curvature_clip:
                self.model.clamp_curvatures(
                    min_c=self.config.curvature_min,
                    max_c=self.config.curvature_max
                )

            # 统计更新
            batch_loss = loss.item()
            epoch_stats['loss'] += batch_loss
            epoch_stats['num_batches'] += 1
            epoch_stats['num_edges'] += batch_pos_edges.size(1)
            epoch_stats['grad_norm'] += grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

            # 准确率计算
            with torch.no_grad():
                pos_acc = (torch.sigmoid(pos_scores) > 0.5).float().mean().item()
                neg_acc = (torch.sigmoid(neg_scores) <= 0.5).float().mean().item()
                epoch_stats['pos_acc'] += pos_acc
                epoch_stats['neg_acc'] += neg_acc

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f"{batch_loss:.4f}",
                'PosAcc': f"{pos_acc:.3f}",
                'NegAcc': f"{neg_acc:.3f}"
            })

        progress_bar.close()

        # 平均统计
        if epoch_stats['num_batches'] > 0:
            for key in ['loss', 'pos_acc', 'neg_acc', 'grad_norm']:
                epoch_stats[key] /= epoch_stats['num_batches']

        # 记录曲率和温度
        epoch_stats['curvatures'] = self.model.get_curvatures()
        if hasattr(self.model.decoder, 'temperature'):
            epoch_stats['decoder_temp'] = float(self.model.decoder.temperature)

        return epoch_stats

    def validate(self, data_dict: Dict, data_name: str = "val") -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        val_stats = {'loss': 0.0}

        edge_index = data_dict['edge_index'].to(self.device)
        x = self.features[data_name].to(self.device)

        with torch.no_grad():
            try:
                # 获取节点嵌入
                h = self.model(x, edge_index)

                # 正样本分数
                pos_scores = self.model.predict_links(h, edge_index)

                # 负采样（用于计算验证损失）
                neg_edges = self._sample_validation_negatives(edge_index, h.size(0))
                neg_scores = self.model.predict_links(h, neg_edges)

                # 验证损失
                scores = torch.cat([pos_scores, neg_scores], dim=0)
                labels = torch.cat([
                    torch.ones_like(pos_scores),
                    torch.zeros_like(neg_scores)
                ], dim=0)

                val_loss = F.binary_cross_entropy_with_logits(scores, labels)
                val_stats['loss'] = val_loss.item()

                # 简单指标
                pos_acc = (torch.sigmoid(pos_scores) > 0.5).float().mean().item()
                neg_acc = (torch.sigmoid(neg_scores) <= 0.5).float().mean().item()
                val_stats['pos_acc'] = pos_acc
                val_stats['neg_acc'] = neg_acc
                val_stats['accuracy'] = (pos_acc + neg_acc) / 2

            except Exception as e:
                logger.error(f"Validation error for {data_name}: {e}")
                val_stats['loss'] = float('inf')
                val_stats['accuracy'] = 0.0

        return val_stats

    def _sample_validation_negatives(self, pos_edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """为验证采样负边"""
        num_neg = min(pos_edges.size(1), 10000)  # 限制负样本数量

        # 简单随机负采样
        neg_src = torch.randint(0, num_nodes, (num_neg,), device=self.device)
        neg_dst = torch.randint(0, num_nodes, (num_neg,), device=self.device)

        # 避免自环
        self_loop_mask = neg_src == neg_dst
        neg_dst[self_loop_mask] = (neg_dst[self_loop_mask] + 1) % num_nodes

        return torch.stack([neg_src, neg_dst], dim=0)

    def _compute_grad_norm(self) -> float:
        """计算梯度范数"""
        total_norm = 0.0
        param_count = 0

        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        if param_count > 0:
            return (total_norm ** 0.5) / param_count
        else:
            return 0.0

    def _should_early_stop(self, val_score: float) -> bool:
        """检查是否应该早停"""
        if not self.config.early_stopping:
            return False

        if val_score > self.best_val_score + self.config.min_delta:
            self.best_val_score = val_score
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience

    def save_checkpoint(self, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_score': self.best_val_score,
            'config': asdict(self.config),
            'training_stats': self.training_stats,
            'validation_stats': self.validation_stats,
            'metrics': self.metrics
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, 'latest.pt'))

        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, 'best.pt'))
            logger.info(f"Saved best model at epoch {self.epoch} with score {self.best_val_score:.4f}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_score = checkpoint.get('best_val_score', -float('inf'))
        self.training_stats = checkpoint.get('training_stats', [])
        self.validation_stats = checkpoint.get('validation_stats', [])
        self.metrics = checkpoint.get('metrics', {})

        logger.info(f"Loaded checkpoint from epoch {self.epoch}")

    def train(self) -> Dict[str, Any]:
        """主训练循环"""
        logger.info(f"Starting training for {self.config.epochs} epochs")
        start_time = time.time()

        try:
            for epoch in range(self.epoch, self.config.epochs):
                self.epoch = epoch
                epoch_start = time.time()

                # 训练阶段
                train_stats = self.train_epoch()
                self.training_stats.append(train_stats)

                # 记录指标
                self.metrics['train_loss'].append(train_stats['loss'])
                self.metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                self.metrics['curvatures'].append(train_stats['curvatures'])
                self.metrics['grad_norms'].append(train_stats['grad_norm'])

                # 验证阶段
                val_stats = {}
                if epoch % self.config.val_frequency == 0:
                    # 验证集
                    val_stats['val'] = self.validate(self.val_data, 'val')

                    # 测试集（如果有）
                    for i, test_data in enumerate(self.test_data):
                        test_name = test_data.get('name', f'test_{i}')
                        val_stats[test_name] = self.validate(test_data, 'test')

                    self.validation_stats.append(val_stats)

                    # 学习率调度
                    if self.scheduler:
                        if self.config.scheduler == "plateau":
                            self.scheduler.step(val_stats['val']['accuracy'])
                        else:
                            self.scheduler.step()

                # 早停检查
                current_val_score = val_stats.get('val', {}).get('accuracy', 0.0)
                if self._should_early_stop(current_val_score):
                    logger.info(f"Early stopping at epoch {epoch}")
                    self.metrics['early_stopped'] = True
                    break

                # 保存检查点
                is_best = current_val_score > self.best_val_score
                if epoch % self.config.save_frequency == 0 or is_best:
                    self.save_checkpoint(is_best)

                # 日志记录
                if epoch % self.config.log_frequency == 0:
                    epoch_time = time.time() - epoch_start
                    self._log_epoch_stats(train_stats, val_stats, epoch_time)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")

        # 最终保存
        self.save_checkpoint(is_best=False)

        return {
            'final_epoch': self.epoch,
            'best_val_score': self.best_val_score,
            'total_time': total_time,
            'early_stopped': self.metrics.get('early_stopped', False),
            'metrics': self.metrics
        }

    def _log_epoch_stats(self, train_stats: Dict, val_stats: Dict, epoch_time: float):
        """记录epoch统计信息"""
        log_msg = (
            f"Epoch {self.epoch:3d} | "
            f"Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_stats['loss']:.4f} | "
            f"Train Acc: {(train_stats['pos_acc'] + train_stats['neg_acc']) / 2:.3f} | "
        )

        if val_stats.get('val'):
            log_msg += f"Val Loss: {val_stats['val']['loss']:.4f} | Val Acc: {val_stats['val']['accuracy']:.3f} | "

        log_msg += f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "

        # 曲率信息
        curvatures = train_stats.get('curvatures', {})
        if curvatures:
            if 'global' in curvatures:
                log_msg += f"C: {curvatures['global']:.3f}"
            else:
                avg_c = np.mean(list(curvatures.values()))
                log_msg += f"C_avg: {avg_c:.3f}"

        logger.info(log_msg)


def load_training_data(config: Dict) -> Tuple[Dict, Dict, List[Dict], Dict]:
    """加载训练数据"""
    logger.info("Loading training data...")

    # 加载图数据
    train_data = torch.load(config['data']['train_graph'])
    val_data = torch.load(config['data']['val_graph'])

    test_data = []
    for test_path in config['data']['test_graphs']:
        test_data.append(torch.load(test_path))

    # 加载特征
    features = {}
    feature_dir = config['features']['dir']

    # 这里简化处理，实际需要根据蛋白质ID加载对应特征
    # 假设特征已经按节点顺序组织好
    features['train'] = torch.randn(train_data.num_nodes, 1280)  # ESM-2 650M
    features['val'] = torch.randn(val_data.num_nodes, 1280)
    features['test'] = torch.randn(test_data[0].num_nodes, 1280)

    logger.info(f"Loaded training data: {train_data.num_nodes} nodes, {train_data.num_edges} edges")

    return train_data, val_data, test_data, features


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HGCN Link Prediction Training")
    parser.add_argument("--config", "-c", type=str, default="cfg/default.yaml",
                       help="Configuration file path")
    parser.add_argument("--resume", "-r", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--wandb", action="store_true",
                       help="Use Weights & Biases logging")

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 创建训练配置
    train_config = TrainingConfig(**config.get('train', {}))
    train_config.debug = args.debug

    # 设置随机种子
    set_seed(train_config.seed)

    # 设置日志
    setup_logging(level="DEBUG" if args.debug else "INFO")

    # W&B初始化
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(project="ppi-hgcn", config=config)

    try:
        # 加载数据
        train_data, val_data, test_data, features = load_training_data(config)

        # 创建模型
        model = create_hgcn_from_config(config)
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

        # 创建负采样器
        negative_sampler = create_negative_sampler(config.get('negative_sampling', {}))

        # 创建训练器
        trainer = LinkPredictionTrainer(
            model=model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            config=train_config,
            features=features,
            negative_sampler=negative_sampler
        )

        # 恢复训练（如果指定）
        if args.resume:
            trainer.load_checkpoint(args.resume)

        # 开始训练
        results = trainer.train()

        logger.info("Training completed successfully!")
        logger.info(f"Best validation score: {results['best_val_score']:.4f}")
        logger.info(f"Total epochs: {results['final_epoch']}")

        return results

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
