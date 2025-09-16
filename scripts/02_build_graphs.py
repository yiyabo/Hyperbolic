#!/usr/bin/env python3
"""
图数据构建脚本

该脚本用于构建和处理图数据，包括：
- 加载和预处理HuRI数据
- 加载和预处理STRING数据
- 统一ID映射
- 数据拆分（冷蛋白拆分）
- 生成最终的图数据文件
"""

import argparse
import logging
import yaml
import sys
import json
from pathlib import Path
import pandas as pd
import torch
from typing import Dict, List, Tuple
import numpy as np

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataio.load_huri import HuRILoader
from dataio.load_string import STRINGLoader


def setup_logging(log_file: str = "logs/build_graphs.log", level: str = "INFO"):
    """设置日志"""
    # 创建logs目录
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # 配置日志格式
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_cold_protein_split(graph_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    创建冷蛋白拆分

    Args:
        graph_data: 图数据对象
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        Dict: 包含不同拆分的边索引
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating cold protein split...")

    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    edge_index = graph_data.edge_index
    num_edges = edge_index.size(1) // 2  # 无向图，实际边数为一半

    # 获取所有唯一的边（去除双向重复）
    edges_set = set()
    unique_edges = []

    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        edge = tuple(sorted([u, v]))
        if edge not in edges_set:
            edges_set.add(edge)
            unique_edges.append([u, v])

    logger.info(f"Total unique edges: {len(unique_edges)}")

    # 随机打乱边
    unique_edges = np.array(unique_edges)
    perm = np.random.permutation(len(unique_edges))
    unique_edges = unique_edges[perm]

    # 获取所有节点
    all_nodes = set(range(graph_data.num_nodes))

    # 计算拆分点
    n_train = int(len(unique_edges) * train_ratio)
    n_val = int(len(unique_edges) * val_ratio)

    # 拆分边
    train_edges = unique_edges[:n_train]
    val_edges = unique_edges[n_train:n_train + n_val]
    test_edges = unique_edges[n_train + n_val:]

    logger.info(f"Edge split - Train: {len(train_edges)}, Val: {len(val_edges)}, Test: {len(test_edges)}")

    # 获取每个拆分中涉及的节点
    train_nodes = set()
    for edge in train_edges:
        train_nodes.update(edge)

    val_nodes = set()
    for edge in val_edges:
        val_nodes.update(edge)

    test_nodes = set()
    for edge in test_edges:
        test_nodes.update(edge)

    # 冷蛋白约束：测试集和验证集中的节点不能在训练集中出现
    # 重新分配节点以满足冷蛋白约束
    logger.info("Applying cold protein constraints...")

    # 找到冲突的节点
    val_conflict_nodes = val_nodes & train_nodes
    test_conflict_nodes = test_nodes & train_nodes

    logger.info(f"Conflicting nodes - Val: {len(val_conflict_nodes)}, Test: {len(test_conflict_nodes)}")

    # 简化策略：将包含冲突节点的边移到训练集
    final_train_edges = list(train_edges)
    final_val_edges = []
    final_test_edges = []

    # 处理验证集
    for edge in val_edges:
        if edge[0] in train_nodes or edge[1] in train_nodes:
            final_train_edges.append(edge)
        else:
            final_val_edges.append(edge)

    # 处理测试集
    current_train_val_nodes = set()
    for edge in final_train_edges + final_val_edges:
        current_train_val_nodes.update(edge)

    for edge in test_edges:
        if edge[0] in current_train_val_nodes or edge[1] in current_train_val_nodes:
            final_train_edges.append(edge)
        else:
            final_test_edges.append(edge)

    logger.info(f"Final edge split - Train: {len(final_train_edges)}, Val: {len(final_val_edges)}, Test: {len(final_test_edges)}")

    # 转换为PyTorch张量（包含双向边）
    def edges_to_tensor(edges_list):
        if not edges_list:
            return torch.empty((2, 0), dtype=torch.long)

        # 添加双向边
        all_edges = []
        for edge in edges_list:
            all_edges.extend([[edge[0], edge[1]], [edge[1], edge[0]]])

        return torch.tensor(all_edges, dtype=torch.long).t().contiguous()

    train_edge_index = edges_to_tensor(final_train_edges)
    val_edge_index = edges_to_tensor(final_val_edges)
    test_edge_index = edges_to_tensor(final_test_edges)

    # 验证冷蛋白约束
    train_nodes_final = set(train_edge_index.flatten().tolist()) if train_edge_index.size(1) > 0 else set()
    val_nodes_final = set(val_edge_index.flatten().tolist()) if val_edge_index.size(1) > 0 else set()
    test_nodes_final = set(test_edge_index.flatten().tolist()) if test_edge_index.size(1) > 0 else set()

    val_violations = len(val_nodes_final & train_nodes_final)
    test_violations = len(test_nodes_final & (train_nodes_final | val_nodes_final))

    logger.info(f"Cold protein violations - Val: {val_violations}, Test: {test_violations}")

    if val_violations > 0 or test_violations > 0:
        logger.warning("Cold protein constraints not perfectly satisfied")
    else:
        logger.info("Cold protein constraints satisfied")

    return {
        'train': train_edge_index,
        'val': val_edge_index,
        'test': test_edge_index,
        'train_nodes': train_nodes_final,
        'val_nodes': val_nodes_final,
        'test_nodes': test_nodes_final
    }


def process_huri_data(config_file: str, output_dir: str) -> bool:
    """处理HuRI数据"""
    logger = logging.getLogger(__name__)
    logger.info("Processing HuRI data...")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # 创建HuRI加载器
        loader = HuRILoader(config)

        # 加载和处理数据
        graph_data = loader.load_and_process()

        # 创建冷蛋白拆分
        split_config = config.get('split', {})
        splits = create_cold_protein_split(
            graph_data,
            train_ratio=split_config.get('train_ratio', 0.8),
            val_ratio=split_config.get('val_ratio', 0.1),
            test_ratio=split_config.get('test_ratio', 0.1),
            seed=split_config.get('random_seed', 42)
        )

        # 将拆分信息添加到图数据
        graph_data.train_edge_index = splits['train']
        graph_data.val_edge_index = splits['val']
        graph_data.test_edge_index = splits['test']
        graph_data.train_nodes = splits['train_nodes']
        graph_data.val_nodes = splits['val_nodes']
        graph_data.test_nodes = splits['test_nodes']

        # 保存处理后的数据
        output_file = f"{output_dir}/huri_graph.pt"
        mapping_file = f"{output_dir}/huri_node_mapping.json"

        loader.save_processed_data(output_file, mapping_file)

        # 保存拆分统计信息
        split_stats = {
            'train_edges': int(splits['train'].size(1)) if splits['train'].size(1) > 0 else 0,
            'val_edges': int(splits['val'].size(1)) if splits['val'].size(1) > 0 else 0,
            'test_edges': int(splits['test'].size(1)) if splits['test'].size(1) > 0 else 0,
            'train_nodes': len(splits['train_nodes']),
            'val_nodes': len(splits['val_nodes']),
            'test_nodes': len(splits['test_nodes']),
            'total_nodes': graph_data.num_nodes,
            'total_edges': graph_data.num_edges
        }

        with open(f"{output_dir}/huri_split_stats.json", 'w') as f:
            json.dump(split_stats, f, indent=2)

        logger.info(f"HuRI data processed successfully: {output_file}")
        logger.info(f"Split statistics: {split_stats}")

        return True

    except Exception as e:
        logger.error(f"Error processing HuRI data: {e}")
        return False


def process_string_data(config_file: str, output_dir: str) -> bool:
    """处理STRING数据"""
    logger = logging.getLogger(__name__)

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        species_name = config['species']['common_name']
        logger.info(f"Processing STRING {species_name} data...")

        # 创建STRING加载器
        loader = STRINGLoader(config)

        # 加载和处理数据
        graph_data = loader.load_and_process()

        # 为不同置信度阈值创建子图
        confidence_subgraphs = loader.create_confidence_subgraphs()

        # 保存主图数据
        output_file = f"{output_dir}/string_{species_name}_graph.pt"
        mapping_file = f"{output_dir}/string_{species_name}_node_mapping.json"

        loader.save_processed_data(output_file, mapping_file)

        # 保存置信度子图
        for threshold, subgraph in confidence_subgraphs.items():
            subgraph_file = f"{output_dir}/string_{species_name}_conf{threshold}_graph.pt"
            torch.save(subgraph, subgraph_file)
            logger.info(f"Saved confidence {threshold} subgraph: {subgraph_file}")

        logger.info(f"STRING {species_name} data processed successfully: {output_file}")

        return True

    except Exception as e:
        logger.error(f"Error processing STRING data: {e}")
        return False


def validate_graph_data(graph_file: str) -> bool:
    """验证图数据文件"""
    logger = logging.getLogger(__name__)

    try:
        graph_data = torch.load(graph_file, map_location='cpu')

        # 基本验证
        assert hasattr(graph_data, 'edge_index'), "Missing edge_index"
        assert hasattr(graph_data, 'num_nodes'), "Missing num_nodes"

        edge_index = graph_data.edge_index
        num_nodes = graph_data.num_nodes

        # 验证边索引
        assert edge_index.dim() == 2, f"Invalid edge_index dimension: {edge_index.dim()}"
        assert edge_index.size(0) == 2, f"Invalid edge_index shape: {edge_index.shape}"

        # 验证节点索引范围
        max_node = edge_index.max().item()
        assert max_node < num_nodes, f"Node index {max_node} exceeds num_nodes {num_nodes}"

        logger.info(f"Graph validation passed: {graph_file}")
        logger.info(f"  Nodes: {num_nodes}, Edges: {edge_index.size(1)}")

        return True

    except Exception as e:
        logger.error(f"Graph validation failed for {graph_file}: {e}")
        return False


def build_all_graphs(config_dir: str = "cfg", output_dir: str = "data/processed"):
    """构建所有图数据"""
    logger = logging.getLogger(__name__)
    logger.info("Building all graph data...")

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 配置文件列表
    config_files = [
        ("huri", f"{config_dir}/data_huri.yaml", process_huri_data),
        ("string_human", f"{config_dir}/data_string_human.yaml", process_string_data),
        ("string_mouse", f"{config_dir}/data_string_mouse.yaml", process_string_data),
        ("string_yeast", f"{config_dir}/data_string_yeast.yaml", process_string_data)
    ]

    results = {}

    for name, config_file, process_func in config_files:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {name}")
        logger.info(f"{'='*50}")

        if not Path(config_file).exists():
            logger.warning(f"Config file not found: {config_file}, skipping...")
            results[name] = False
            continue

        try:
            success = process_func(config_file, output_dir)
            results[name] = success

        except Exception as e:
            logger.error(f"Error processing {name}: {e}")
            results[name] = False

    # 验证生成的图文件
    logger.info(f"\n{'='*50}")
    logger.info("Validating generated graphs")
    logger.info(f"{'='*50}")

    graph_files = list(Path(output_dir).glob("*.pt"))
    validation_results = {}

    for graph_file in graph_files:
        file_name = graph_file.name
        is_valid = validate_graph_data(str(graph_file))
        validation_results[file_name] = is_valid

    # 汇总结果
    logger.info(f"\n{'='*50}")
    logger.info("Graph Building Summary")
    logger.info(f"{'='*50}")

    total_tasks = len(results)
    successful_tasks = sum(results.values())

    logger.info("Processing Results:")
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {name:15s}: {status}")

    logger.info("\nValidation Results:")
    total_graphs = len(validation_results)
    valid_graphs = sum(validation_results.values())

    for file_name, is_valid in validation_results.items():
        status = "✓ VALID" if is_valid else "✗ INVALID"
        logger.info(f"  {file_name:30s}: {status}")

    logger.info(f"\nSummary:")
    logger.info(f"  Processing: {successful_tasks}/{total_tasks} tasks completed successfully")
    logger.info(f"  Validation: {valid_graphs}/{total_graphs} graphs are valid")

    # 创建完成标记
    all_success = (successful_tasks == total_tasks and valid_graphs == total_graphs)
    if all_success:
        with open(f"{output_dir}/.build_complete", 'w') as f:
            f.write(f"All graphs built successfully at {pd.Timestamp.now()}\n")
        logger.info("All graphs built and validated successfully!")
        return True
    else:
        logger.error("Some graphs failed to build or validate. Check the logs for details.")
        return False


def main():
    parser = argparse.ArgumentParser(description="图数据构建脚本")
    parser.add_argument("--config-dir", "-c", default="cfg",
                       help="配置文件目录 (default: cfg)")
    parser.add_argument("--output-dir", "-o", default="data/processed",
                       help="输出目录 (default: data/processed)")
    parser.add_argument("--log-level", "-l", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别 (default: INFO)")
    parser.add_argument("--dataset", "-d",
                       choices=["huri", "string_human", "string_mouse", "string_yeast", "all"],
                       default="all", help="要处理的数据集 (default: all)")
    parser.add_argument("--validate", "-v", action="store_true",
                       help="验证现有的图文件")

    args = parser.parse_args()

    # 设置日志
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting graph building script...")
    logger.info(f"Config directory: {args.config_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Dataset: {args.dataset}")

    if args.validate:
        graph_files = list(Path(args.output_dir).glob("*.pt"))
        if not graph_files:
            logger.error(f"No graph files found in {args.output_dir}")
            sys.exit(1)

        all_valid = True
        for graph_file in graph_files:
            is_valid = validate_graph_data(str(graph_file))
            if not is_valid:
                all_valid = False

        sys.exit(0 if all_valid else 1)

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        if args.dataset == "all":
            success = build_all_graphs(args.config_dir, args.output_dir)
        elif args.dataset == "huri":
            config_file = f"{args.config_dir}/data_huri.yaml"
            success = process_huri_data(config_file, args.output_dir)
        else:
            config_file = f"{args.config_dir}/data_{args.dataset}.yaml"
            success = process_string_data(config_file, args.output_dir)

        if success:
            logger.info("Graph building completed successfully!")
            sys.exit(0)
        else:
            logger.error("Graph building failed!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
