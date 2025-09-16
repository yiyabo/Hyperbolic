"""
UniProt ID映射模块

该模块负责通过UniProt API进行ID映射，包括：
- 批量ID映射请求
- 断点续传和重试机制
- 支持多种ID类型转换
- 缓存映射结果
"""

import requests
import pandas as pd
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import pickle
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class UniProtIDMapper:
    """UniProt ID映射器"""

    def __init__(self, cache_dir: str = "data/raw/mapping", retry_limit: int = 3, batch_size: int = 1000):
        """
        初始化UniProt ID映射器

        Args:
            cache_dir: 缓存目录
            retry_limit: 重试次数限制
            batch_size: 批处理大小
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retry_limit = retry_limit
        self.batch_size = batch_size

        # UniProt API URLs
        self.mapping_url = "https://rest.uniprot.org/idmapping"
        self.run_url = f"{self.mapping_url}/run"
        self.status_url = f"{self.mapping_url}/status"
        self.results_url = f"{self.mapping_url}/results"

        # 支持的ID类型映射
        self.supported_id_types = {
            'ensembl_gene': 'Ensembl',
            'ensembl_protein': 'Ensembl_PRO',
            'string_protein': 'STRING',
            'uniprotkb': 'UniProtKB',
            'uniprotkb_id': 'UniProtKB-ID',
            'gene_name': 'Gene_Name',
            'hgnc': 'HGNC',
            'refseq': 'RefSeq_Protein',
            'pdb': 'PDB',
            'ncbi_taxon': 'NCBI_TaxID'
        }

    def get_cache_path(self, from_db: str, to_db: str, batch_id: str = "default") -> Path:
        """
        获取缓存文件路径

        Args:
            from_db: 源数据库
            to_db: 目标数据库
            batch_id: 批次ID

        Returns:
            Path: 缓存文件路径
        """
        filename = f"{from_db}_to_{to_db}_{batch_id}.json"
        return self.cache_dir / filename

    def load_cache(self, from_db: str, to_db: str, batch_id: str = "default") -> Dict[str, str]:
        """
        加载缓存的映射结果

        Args:
            from_db: 源数据库
            to_db: 目标数据库
            batch_id: 批次ID

        Returns:
            Dict[str, str]: 缓存的映射字典
        """
        cache_path = self.get_cache_path(from_db, to_db, batch_id)

        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached_mapping = json.load(f)
                logger.info(f"Loaded {len(cached_mapping)} cached mappings from {cache_path}")
                return cached_mapping
            except Exception as e:
                logger.warning(f"Error loading cache from {cache_path}: {e}")

        return {}

    def save_cache(self, mapping: Dict[str, str], from_db: str, to_db: str, batch_id: str = "default"):
        """
        保存映射结果到缓存

        Args:
            mapping: 映射字典
            from_db: 源数据库
            to_db: 目标数据库
            batch_id: 批次ID
        """
        cache_path = self.get_cache_path(from_db, to_db, batch_id)

        try:
            with open(cache_path, 'w') as f:
                json.dump(mapping, f, indent=2)
            logger.info(f"Saved {len(mapping)} mappings to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving cache to {cache_path}: {e}")

    def submit_mapping_job(self, ids: List[str], from_db: str, to_db: str) -> Optional[str]:
        """
        提交ID映射任务

        Args:
            ids: 待映射的ID列表
            from_db: 源数据库
            to_db: 目标数据库

        Returns:
            Optional[str]: 任务ID，如果失败返回None
        """
        # 转换数据库名称
        from_db_api = self.supported_id_types.get(from_db, from_db)
        to_db_api = self.supported_id_types.get(to_db, to_db)

        data = {
            'from': from_db_api,
            'to': to_db_api,
            'ids': ','.join(ids)
        }

        for attempt in range(self.retry_limit):
            try:
                logger.info(f"Submitting mapping job (attempt {attempt + 1}/{self.retry_limit})")
                logger.info(f"From: {from_db_api}, To: {to_db_api}, IDs: {len(ids)}")

                response = requests.post(self.run_url, data=data, timeout=30)

                if response.status_code == 200:
                    job_id = response.json().get('jobId')
                    logger.info(f"Job submitted successfully. Job ID: {job_id}")
                    return job_id
                else:
                    logger.warning(f"Job submission failed with status {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.retry_limit}): {e}")

            if attempt < self.retry_limit - 1:
                wait_time = 2 ** attempt  # 指数退避
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

        logger.error("All job submission attempts failed")
        return None

    def check_job_status(self, job_id: str) -> Tuple[str, Optional[str]]:
        """
        检查映射任务状态

        Args:
            job_id: 任务ID

        Returns:
            Tuple[str, Optional[str]]: (状态, 错误信息)
        """
        try:
            status_url = f"{self.status_url}/{job_id}"
            response = requests.get(status_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data.get('jobStatus', 'UNKNOWN'), None
            else:
                return 'ERROR', f"Status check failed: {response.status_code}"

        except requests.exceptions.RequestException as e:
            return 'ERROR', f"Request failed: {e}"

    def get_mapping_results(self, job_id: str) -> Dict[str, str]:
        """
        获取映射结果

        Args:
            job_id: 任务ID

        Returns:
            Dict[str, str]: 映射结果字典
        """
        try:
            results_url = f"{self.results_url}/{job_id}"
            response = requests.get(results_url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                mapping = {}

                for result in data.get('results', []):
                    from_id = result.get('from')
                    to_id = result.get('to')
                    if from_id and to_id:
                        mapping[from_id] = to_id

                logger.info(f"Retrieved {len(mapping)} mappings from job {job_id}")
                return mapping
            else:
                logger.error(f"Failed to get results: {response.status_code} - {response.text}")
                return {}

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {}

    def wait_for_job_completion(self, job_id: str, max_wait_time: int = 300) -> bool:
        """
        等待任务完成

        Args:
            job_id: 任务ID
            max_wait_time: 最大等待时间（秒）

        Returns:
            bool: 是否成功完成
        """
        start_time = time.time()
        wait_interval = 5  # 起始等待间隔

        while time.time() - start_time < max_wait_time:
            status, error = self.check_job_status(job_id)

            if status == 'FINISHED':
                logger.info(f"Job {job_id} completed successfully")
                return True
            elif status in ['ERROR', 'FAILED']:
                logger.error(f"Job {job_id} failed: {error}")
                return False
            elif status == 'RUNNING':
                logger.info(f"Job {job_id} is still running...")
            else:
                logger.warning(f"Unknown job status: {status}")

            # 动态调整等待间隔
            time.sleep(wait_interval)
            wait_interval = min(wait_interval * 1.2, 30)  # 最大30秒

        logger.error(f"Job {job_id} timed out after {max_wait_time} seconds")
        return False

    def map_ids_batch(self, ids: List[str], from_db: str, to_db: str) -> Dict[str, str]:
        """
        批量映射ID

        Args:
            ids: 待映射的ID列表
            from_db: 源数据库类型
            to_db: 目标数据库类型

        Returns:
            Dict[str, str]: 映射结果字典
        """
        if not ids:
            return {}

        logger.info(f"Mapping {len(ids)} IDs from {from_db} to {to_db}")

        # 提交任务
        job_id = self.submit_mapping_job(ids, from_db, to_db)
        if not job_id:
            logger.error("Failed to submit mapping job")
            return {}

        # 等待完成
        if not self.wait_for_job_completion(job_id):
            logger.error("Job did not complete successfully")
            return {}

        # 获取结果
        mapping = self.get_mapping_results(job_id)
        return mapping

    def map_ids_with_cache(self, ids: List[str], from_db: str, to_db: str,
                          batch_id: str = "default") -> Dict[str, str]:
        """
        带缓存的ID映射

        Args:
            ids: 待映射的ID列表
            from_db: 源数据库类型
            to_db: 目标数据库类型
            batch_id: 批次ID

        Returns:
            Dict[str, str]: 映射结果字典
        """
        # 加载缓存
        cached_mapping = self.load_cache(from_db, to_db, batch_id)

        # 找出未缓存的ID
        ids_set = set(ids)
        cached_ids = set(cached_mapping.keys())
        uncached_ids = list(ids_set - cached_ids)

        if not uncached_ids:
            logger.info("All IDs found in cache")
            return {id_: cached_mapping[id_] for id_ in ids if id_ in cached_mapping}

        logger.info(f"Found {len(cached_ids & ids_set)} cached mappings, need to map {len(uncached_ids)} new IDs")

        # 映射未缓存的ID
        new_mapping = self.map_ids_batch(uncached_ids, from_db, to_db)

        # 合并结果
        combined_mapping = {**cached_mapping, **new_mapping}

        # 保存到缓存
        if new_mapping:
            self.save_cache(combined_mapping, from_db, to_db, batch_id)

        # 返回请求ID的映射
        return {id_: combined_mapping[id_] for id_ in ids if id_ in combined_mapping}

    def map_ids_in_batches(self, ids: List[str], from_db: str, to_db: str,
                          progress_callback: Optional[callable] = None) -> Dict[str, str]:
        """
        分批映射大量ID

        Args:
            ids: 待映射的ID列表
            from_db: 源数据库类型
            to_db: 目标数据库类型
            progress_callback: 进度回调函数

        Returns:
            Dict[str, str]: 映射结果字典
        """
        logger.info(f"Mapping {len(ids)} IDs in batches of {self.batch_size}")

        all_mapping = {}
        total_batches = (len(ids) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(ids), self.batch_size):
            batch_ids = ids[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_ids)} IDs)")

            batch_mapping = self.map_ids_with_cache(
                batch_ids, from_db, to_db,
                batch_id=f"batch_{batch_num}"
            )

            all_mapping.update(batch_mapping)

            if progress_callback:
                progress = batch_num / total_batches
                progress_callback(progress, batch_num, total_batches)

            # 批次间等待，避免过于频繁的请求
            if batch_num < total_batches:
                time.sleep(1)

        logger.info(f"Completed mapping: {len(all_mapping)} successful mappings out of {len(ids)} input IDs")
        return all_mapping

    def create_mapping_file(self, ids: List[str], from_db: str, to_db: str,
                           output_path: str, format: str = 'tsv') -> bool:
        """
        创建映射文件

        Args:
            ids: 待映射的ID列表
            from_db: 源数据库类型
            to_db: 目标数据库类型
            output_path: 输出文件路径
            format: 输出格式 ('tsv', 'json', 'csv')

        Returns:
            bool: 是否成功创建文件
        """
        try:
            mapping = self.map_ids_in_batches(ids, from_db, to_db)

            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(mapping, f, indent=2)

            elif format.lower() in ['tsv', 'csv']:
                separator = '\t' if format.lower() == 'tsv' else ','
                df = pd.DataFrame([
                    {from_db: from_id, to_db: to_id}
                    for from_id, to_id in mapping.items()
                ])
                df.to_csv(output_path, sep=separator, index=False)

            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Mapping file created: {output_path} ({len(mapping)} mappings)")
            return True

        except Exception as e:
            logger.error(f"Error creating mapping file: {e}")
            return False

    def get_mapping_statistics(self, mapping: Dict[str, str],
                             original_ids: List[str]) -> Dict[str, any]:
        """
        获取映射统计信息

        Args:
            mapping: 映射结果
            original_ids: 原始ID列表

        Returns:
            Dict: 统计信息
        """
        total_ids = len(original_ids)
        mapped_ids = len(mapping)
        unique_targets = len(set(mapping.values()))

        stats = {
            'total_input_ids': total_ids,
            'successfully_mapped': mapped_ids,
            'mapping_rate': mapped_ids / total_ids if total_ids > 0 else 0,
            'unique_target_ids': unique_targets,
            'redundancy_rate': (mapped_ids - unique_targets) / mapped_ids if mapped_ids > 0 else 0,
            'unmapped_ids': total_ids - mapped_ids
        }

        return stats

    def validate_id_format(self, ids: List[str], id_type: str) -> Tuple[List[str], List[str]]:
        """
        验证ID格式

        Args:
            ids: ID列表
            id_type: ID类型

        Returns:
            Tuple[List[str], List[str]]: (有效ID列表, 无效ID列表)
        """
        valid_ids = []
        invalid_ids = []

        for id_ in ids:
            if self._is_valid_id(id_, id_type):
                valid_ids.append(id_)
            else:
                invalid_ids.append(id_)

        if invalid_ids:
            logger.warning(f"Found {len(invalid_ids)} invalid IDs for type {id_type}")

        return valid_ids, invalid_ids

    def _is_valid_id(self, id_: str, id_type: str) -> bool:
        """
        检查ID格式是否有效

        Args:
            id_: ID字符串
            id_type: ID类型

        Returns:
            bool: 是否有效
        """
        if not id_ or not isinstance(id_, str):
            return False

        id_ = id_.strip()
        if not id_:
            return False

        # 基本格式检查
        if id_type == 'ensembl_gene':
            return id_.startswith('ENSG')
        elif id_type == 'ensembl_protein':
            return id_.startswith('ENSP')
        elif id_type == 'uniprotkb':
            return len(id_) >= 6 and ('_' in id_ or len(id_) <= 10)
        elif id_type == 'string_protein':
            return '.' in id_ and len(id_) > 5
        else:
            return True  # 对未知类型不做严格检查


def create_id_mapping(source_ids: List[str], from_db: str, to_db: str,
                     output_path: str, cache_dir: str = "data/raw/mapping") -> bool:
    """
    便捷函数：创建ID映射文件

    Args:
        source_ids: 源ID列表
        from_db: 源数据库类型
        to_db: 目标数据库类型
        output_path: 输出文件路径
        cache_dir: 缓存目录

    Returns:
        bool: 是否成功
    """
    mapper = UniProtIDMapper(cache_dir=cache_dir)
    return mapper.create_mapping_file(source_ids, from_db, to_db, output_path)


if __name__ == "__main__":
    # 示例用法
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 测试映射
    mapper = UniProtIDMapper()

    test_ids = [
        "ENSG00000139618",  # BRCA2
        "ENSG00000012048",  # BRCA1
        "ENSG00000141510"   # TP53
    ]

    print("Testing ID mapping...")
    mapping = mapper.map_ids_with_cache(test_ids, 'ensembl_gene', 'uniprotkb')

    print(f"\nMapping results:")
    for source_id, target_id in mapping.items():
        print(f"  {source_id} -> {target_id}")

    # 统计信息
    stats = mapper.get_mapping_statistics(mapping, test_ids)
    print(f"\nMapping statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
