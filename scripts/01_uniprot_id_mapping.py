#!/usr/bin/env python3
"""
UniProt ID映射脚本

该脚本用于创建各种数据源到UniProt ID的映射文件，支持：
- HuRI (Ensembl Gene ID -> UniProt)
- STRING (STRING Protein ID -> UniProt)
- 批量处理和缓存
- 断点续传
"""

import argparse
import logging
import yaml
import sys
from pathlib import Path
import pandas as pd
import json
from typing import Dict, Optional

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataio.id_mapping_uniprot import UniProtIDMapper


def setup_logging(log_file: str = "logs/id_mapping.log", level: str = "INFO"):
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


def extract_ids_from_huri(huri_file: str) -> list:
    """从HuRI文件中提取Ensembl Gene ID"""
    logger = logging.getLogger(__name__)
    logger.info(f"Extracting IDs from HuRI file: {huri_file}")

    try:
        # 尝试不同的分隔符和列名
        for sep in ['\t', ',']:
            try:
                df = pd.read_csv(huri_file, sep=sep, nrows=5)
                break
            except:
                continue
        else:
            raise ValueError("Cannot read HuRI file with common separators")

        # 重新读取完整文件
        df = pd.read_csv(huri_file, sep=sep)

        # 寻找蛋白质ID列
        protein_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['protein', 'gene', 'ensembl', 'interactor']):
                protein_cols.append(col)

        if len(protein_cols) >= 2:
            # 使用前两列作为蛋白质ID
            ids = set()
            ids.update(df[protein_cols[0]].dropna().astype(str))
            ids.update(df[protein_cols[1]].dropna().astype(str))
        else:
            # 假设前两列是蛋白质ID
            ids = set()
            ids.update(df.iloc[:, 0].dropna().astype(str))
            ids.update(df.iloc[:, 1].dropna().astype(str))

        ids = [id_ for id_ in ids if id_ != 'nan']
        logger.info(f"Extracted {len(ids)} unique IDs from HuRI")
        return ids

    except Exception as e:
        logger.error(f"Error extracting IDs from HuRI: {e}")
        return []


def extract_ids_from_string_aliases(aliases_file: str, taxid: int) -> list:
    """从STRING aliases文件中提取STRING protein ID"""
    logger = logging.getLogger(__name__)
    logger.info(f"Extracting STRING protein IDs from: {aliases_file}")

    try:
        # 读取STRING aliases文件
        df = pd.read_csv(aliases_file, sep='\t', compression='gzip')

        # 过滤指定物种
        if 'species' in df.columns:
            df = df[df['species'] == taxid]

        # 提取STRING protein ID
        if 'string_protein_id' in df.columns:
            ids = df['string_protein_id'].dropna().unique().tolist()
        elif df.columns[0]:  # 假设第一列是string_protein_id
            ids = df.iloc[:, 0].dropna().unique().tolist()
        else:
            raise ValueError("Cannot find STRING protein ID column")

        logger.info(f"Extracted {len(ids)} unique STRING protein IDs")
        return ids

    except Exception as e:
        logger.error(f"Error extracting STRING IDs: {e}")
        return []


def create_huri_mapping(
    config_file: str,
    output_dir: str,
    offline: bool = False,
    local_mapping_path: Optional[str] = None,
):
    """创建HuRI ID映射"""
    logger = logging.getLogger(__name__)
    logger.info("Creating HuRI ID mapping...")

    # 读取配置
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # 获取HuRI文件路径
    huri_file = config['raw_data']['interactions']

    # 检查文件是否存在
    if not Path(huri_file).exists():
        logger.error(f"HuRI file not found: {huri_file}")
        return False

    # 提取ID
    ensembl_ids = extract_ids_from_huri(huri_file)
    if not ensembl_ids:
        logger.error("No IDs extracted from HuRI")
        return False

    mapper = UniProtIDMapper(cache_dir=f"{output_dir}/cache")
    output_file = f"{output_dir}/huri_uniprot_mapping.tsv"

    if offline and local_mapping_path:
        local_path = Path(local_mapping_path)
        if not local_path.exists():
            logger.error(
                "Offline mode enabled but local mapping file not found: %s",
                local_mapping_path,
            )
            return False

        logger.info(f"Using local HuRI mapping file: {local_mapping_path}")

        try:
            local_df = pd.read_csv(local_path, sep='\t', header=None)
        except Exception:
            local_df = pd.read_csv(local_path, sep='\t')

        if local_df.shape[1] < 2:
            logger.error(
                "Local mapping file %s must have at least two columns", local_mapping_path
            )
            return False

        col_0, col_1 = local_df.columns[:2]
        normalized = local_df.rename(columns={col_0: 'ensembl_gene', col_1: 'uniprotkb'})
        normalized = normalized.dropna(subset=['ensembl_gene', 'uniprotkb'])

        filtered = normalized[normalized['ensembl_gene'].isin(ensembl_ids)].copy()
        if filtered.empty:
            logger.error("No matching Ensembl IDs found in local mapping file")
            return False

        # 去重：同一 Ensembl 只保留第一条
        filtered = filtered.drop_duplicates(subset=['ensembl_gene'])

        filtered.to_csv(output_file, sep='\t', index=False)
        mapping = dict(zip(filtered['ensembl_gene'], filtered['uniprotkb']))

        stats = mapper.get_mapping_statistics(mapping, ensembl_ids)
        stats_file = f"{output_dir}/huri_mapping_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("HuRI mapping statistics (offline):")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")

        logger.info(f"HuRI mapping completed successfully: {output_file}")
        return True

    # 在线模式，调用UniProt API
    def progress_callback(progress, batch_num, total_batches):
        logger.info(f"HuRI mapping progress: {progress:.1%} (batch {batch_num}/{total_batches})")

    mapping = mapper.map_ids_in_batches(
        ensembl_ids,
        'ensembl_gene',
        'uniprotkb',
        progress_callback=progress_callback
    )

    success = mapper.create_mapping_file(
        ensembl_ids,
        'ensembl_gene',
        'uniprotkb',
        output_file,
        format='tsv'
    )

    if success:
        stats = mapper.get_mapping_statistics(mapping, ensembl_ids)
        logger.info("HuRI mapping statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")

        # 保存统计信息
        stats_file = f"{output_dir}/huri_mapping_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"HuRI mapping completed successfully: {output_file}")
        return True
    else:
        logger.error("Failed to create HuRI mapping file")
        return False


def _build_string_mapping_from_aliases(
    aliases_file: str,
    expected_ids: Optional[set] = None,
) -> pd.DataFrame:
    """从STRING aliases生成 String->UniProt 映射"""
    logger = logging.getLogger(__name__)
    logger.info(f"Building STRING mapping from aliases: {aliases_file}")

    aliases_df = pd.read_csv(aliases_file, sep='\t', compression='gzip')

    if 'string_protein_id' not in aliases_df.columns:
        first_col = aliases_df.columns[0]
        aliases_df = aliases_df.rename(columns={first_col: 'string_protein_id'})

    if 'alias' not in aliases_df.columns:
        second_col = aliases_df.columns[1]
        aliases_df = aliases_df.rename(columns={second_col: 'alias'})

    if 'source' not in aliases_df.columns:
        logger.error("STRING aliases file must contain 'source' column for offline mapping")
        return pd.DataFrame(columns=['string_protein_id', 'uniprotkb'])

    mask = aliases_df['source'].str.contains('uniprot', case=False, na=False)
    subset = aliases_df.loc[mask, ['string_protein_id', 'alias']].dropna()
    subset = subset.rename(columns={'alias': 'uniprotkb'})

    if subset.empty:
        logger.warning("No UniProt aliases found in aliases file")
        return subset

    if expected_ids is not None:
        subset = subset[subset['string_protein_id'].isin(expected_ids)]

    subset = subset.drop_duplicates(subset=['string_protein_id'])
    return subset


def create_string_mapping(
    config_file: str,
    output_dir: str,
    offline: bool = False,
):
    """创建STRING ID映射"""
    logger = logging.getLogger(__name__)
    logger.info("Creating STRING ID mapping...")

    # 读取配置
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    species_name = config['species']['common_name']
    taxid = config['species']['taxid']

    # 获取STRING aliases文件路径
    aliases_file = config['raw_data']['protein_aliases']

    # 检查文件是否存在
    if not Path(aliases_file).exists():
        logger.error(f"STRING aliases file not found: {aliases_file}")
        return False

    # 提取STRING protein ID
    string_ids = extract_ids_from_string_aliases(aliases_file, taxid)
    if not string_ids:
        logger.error("No STRING protein IDs extracted")
        return False

    mapper = UniProtIDMapper(cache_dir=f"{output_dir}/cache")
    output_file = f"{output_dir}/string_{species_name}_uniprot_mapping.tsv"

    if offline:
        mapping_df = _build_string_mapping_from_aliases(aliases_file, set(string_ids))

        if mapping_df.empty:
            logger.error("Failed to build STRING mapping from aliases")
            return False

        mapping_dict = dict(zip(mapping_df['string_protein_id'], mapping_df['uniprotkb']))
        mapping_df.to_csv(output_file, sep='\t', index=False)

        stats = mapper.get_mapping_statistics(mapping_dict, string_ids)
        stats['species'] = species_name
        stats['taxid'] = taxid

        logger.info(f"STRING {species_name} mapping statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")

        # 保存统计信息
        stats_file = f"{output_dir}/string_{species_name}_mapping_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"STRING {species_name} mapping completed: {output_file}")
        return True

    # 在线模式，调用UniProt API
    def progress_callback(progress, batch_num, total_batches):
        logger.info(f"STRING {species_name} mapping progress: {progress:.1%} (batch {batch_num}/{total_batches})")

    mapping = mapper.map_ids_in_batches(
        string_ids,
        'string_protein',
        'uniprotkb',
        progress_callback=progress_callback
    )

    success = mapper.create_mapping_file(
        string_ids,
        'string_protein',
        'uniprotkb',
        output_file,
        format='tsv'
    )

    if success:
        stats = mapper.get_mapping_statistics(mapping, string_ids)
        stats['species'] = species_name
        stats['taxid'] = taxid

        logger.info(f"STRING {species_name} mapping statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")

        stats_file = f"{output_dir}/string_{species_name}_mapping_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"STRING {species_name} mapping completed: {output_file}")
        return True

    logger.error(f"Failed to create STRING {species_name} mapping file")
    return False


def create_all_mappings(
    config_dir: str = "cfg",
    output_dir: str = "data/raw/mapping",
    offline: bool = False,
    huri_local_mapping: Optional[str] = None,
):
    """创建所有ID映射"""
    logger = logging.getLogger(__name__)
    logger.info("Creating all ID mappings...")

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 配置文件列表
    config_files = [
        ("huri", f"{config_dir}/data_huri.yaml"),
        ("string_human", f"{config_dir}/data_string_human.yaml"),
        ("string_mouse", f"{config_dir}/data_string_mouse.yaml"),
        ("string_yeast", f"{config_dir}/data_string_yeast.yaml")
    ]

    results = {}

    for name, config_file in config_files:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {name}")
        logger.info(f"{'='*50}")

        if not Path(config_file).exists():
            logger.warning(f"Config file not found: {config_file}, skipping...")
            results[name] = False
            continue

        try:
            if name == "huri":
                success = create_huri_mapping(
                    config_file,
                    output_dir,
                    offline=offline,
                    local_mapping_path=huri_local_mapping,
                )
            else:
                success = create_string_mapping(
                    config_file,
                    output_dir,
                    offline=offline,
                )

            results[name] = success

        except Exception as e:
            logger.error(f"Error processing {name}: {e}")
            results[name] = False

    # 汇总结果
    logger.info(f"\n{'='*50}")
    logger.info("ID Mapping Summary")
    logger.info(f"{'='*50}")

    total_tasks = len(results)
    successful_tasks = sum(results.values())

    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {name:15s}: {status}")

    logger.info(f"\nTotal: {successful_tasks}/{total_tasks} tasks completed successfully")

    # 创建完成标记
    if successful_tasks == total_tasks:
        with open(f"{output_dir}/.mapping_complete", 'w') as f:
            f.write(f"All ID mappings completed successfully at {pd.Timestamp.now()}\n")
        logger.info("All ID mappings completed successfully!")
        return True
    else:
        logger.error("Some ID mappings failed. Check the logs for details.")
        return False


def validate_mappings(mapping_dir: str = "data/raw/mapping"):
    """验证映射文件"""
    logger = logging.getLogger(__name__)
    logger.info("Validating mapping files...")

    mapping_files = [
        "huri_uniprot_mapping.tsv",
        "string_human_uniprot_mapping.tsv",
        "string_mouse_uniprot_mapping.tsv",
        "string_yeast_uniprot_mapping.tsv"
    ]

    all_valid = True

    for filename in mapping_files:
        filepath = Path(mapping_dir) / filename

        if not filepath.exists():
            logger.warning(f"Mapping file not found: {filepath}")
            all_valid = False
            continue

        try:
            df = pd.read_csv(filepath, sep='\t')

            if len(df) == 0:
                logger.error(f"Empty mapping file: {filepath}")
                all_valid = False
                continue

            # 检查列数
            if len(df.columns) != 2:
                logger.error(f"Invalid column count in {filepath}: expected 2, got {len(df.columns)}")
                all_valid = False
                continue

            # 检查空值
            null_count = df.isnull().sum().sum()
            if null_count > 0:
                logger.warning(f"Found {null_count} null values in {filepath}")

            # 基本统计
            logger.info(f"  {filename}: {len(df)} mappings")

        except Exception as e:
            logger.error(f"Error validating {filepath}: {e}")
            all_valid = False

    if all_valid:
        logger.info("All mapping files are valid")
        return True
    else:
        logger.error("Some mapping files have issues")
        return False


def main():
    parser = argparse.ArgumentParser(description="UniProt ID映射脚本")
    parser.add_argument("--config-dir", "-c", default="cfg",
                       help="配置文件目录 (default: cfg)")
    parser.add_argument("--output-dir", "-o", default="data/raw/mapping",
                       help="输出目录 (default: data/raw/mapping)")
    parser.add_argument("--log-level", "-l", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别 (default: INFO)")
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["huri", "string_human", "string_mouse", "string_yeast", "all"],
        default="all",
        help="要处理的数据集 (default: all)",
    )
    parser.add_argument(
        "--validate",
        "-v",
        action="store_true",
        help="验证现有的映射文件",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="优先使用本地文件生成映射（STRING 使用 aliases，HuRI 使用本地映射文件）",
    )
    parser.add_argument(
        "--huri-local-mapping",
        default="data/raw/mapping/ensembl_to_uniprot.tsv",
        help="离线模式下使用的 Ensembl→UniProt 映射文件路径",
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting UniProt ID mapping script...")
    logger.info(f"Config directory: {args.config_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Dataset: {args.dataset}")

    if args.validate:
        success = validate_mappings(args.output_dir)
        sys.exit(0 if success else 1)

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        if args.dataset == "all":
            success = create_all_mappings(
                args.config_dir,
                args.output_dir,
                offline=args.offline,
                huri_local_mapping=args.huri_local_mapping,
            )
        else:
            config_map = {
                "huri": "data_huri.yaml",
                "string_human": "data_string_human.yaml",
                "string_mouse": "data_string_mouse.yaml",
                "string_yeast": "data_string_yeast.yaml",
            }

            config_file = Path(args.config_dir) / config_map[args.dataset]

            if not config_file.exists():
                logger.error(f"Config file not found: {config_file}")
                sys.exit(1)

            if args.dataset == "huri":
                success = create_huri_mapping(
                    str(config_file),
                    args.output_dir,
                    offline=args.offline,
                    local_mapping_path=args.huri_local_mapping,
                )
            else:
                success = create_string_mapping(
                    str(config_file),
                    args.output_dir,
                    offline=args.offline,
                )

        if success:
            logger.info("ID mapping completed successfully!")
            sys.exit(0)
        else:
            logger.error("ID mapping failed!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
