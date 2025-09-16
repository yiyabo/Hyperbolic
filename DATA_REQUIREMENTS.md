# PPI-HGCN 数据需求详细说明

## 📊 数据概览

您的PPI-HGCN项目需要以下几类数据，总存储空间需求约 **50-80GB**

## 🗃️ 详细数据需求

### 1. STRING数据库 (核心数据源)
**用途**: 蛋白质相互作用网络数据
**大小**: ~25-30GB (压缩后 ~8-10GB)

#### 涉及物种:
- **Human (人类)**: taxid=9606
- **Mouse (鼠)**: taxid=10090
- **Yeast (酵母)**: taxid=4932

#### 每个物种需要4类文件:
```bash
# 例如人类 (9606)
9606.protein.physical.links.v12.0.txt.gz    # ~200-500MB 相互作用数据
9606.protein.info.v12.0.txt.gz              # ~50-100MB  蛋白质基本信息
9606.protein.aliases.v12.0.txt.gz           # ~100-200MB ID映射别名
9606.protein.sequences.v12.0.fa.gz          # ~200-800MB 蛋白质序列
```

**详细大小估算**:
- **Human**: ~1.5-2GB (解压后 ~4-6GB)
- **Mouse**: ~1-1.5GB (解压后 ~3-4GB)
- **Yeast**: ~200-500MB (解压后 ~800MB-1.5GB)

### 2. HuRI数据 (Human Reference Interactome)
**用途**: 高质量人类蛋白质相互作用训练集
**大小**: ~100-200MB

```bash
data/raw/huri/
├── HuRI_binaryPPI.tsv          # ~50MB   二元相互作用数据
└── HuRI_sequences.fasta        # ~100MB  对应蛋白质序列
```

**数据规模**:
- 蛋白质数量: ~17,000个
- 相互作用: ~53,000条
- 平均度数: ~6.2

### 3. UniProt ID映射数据
**用途**: 不同蛋白质ID体系间的映射
**大小**: ~500MB-1GB

```bash
data/raw/mapping/
├── huri_uniprot_mapping.tsv    # ~50MB   HuRI到UniProt映射
├── string_uniprot_mapping.tsv  # ~200MB  STRING到UniProt映射
└── id_mapping_cache.pkl        # ~100MB  映射缓存
```

### 4. ESM-2蛋白质特征 (最大数据量)
**用途**: 预训练语言模型提取的蛋白质序列特征
**大小**: ~20-40GB

```bash
features/esm2_650m/
├── human/
│   ├── embeddings_batch_0000.pt     # 每个文件 ~100-500MB
│   ├── embeddings_batch_0001.pt
│   └── ...                          # 总计 ~15-25GB
├── mouse/
│   └── ...                          # ~8-12GB
├── yeast/
│   └── ...                          # ~2-3GB
└── metadata/
    ├── protein_lengths.json         # ~10MB
    └── batch_info.json              # ~5MB
```

**特征维度**: 1280维 (ESM-2 650M模型)
**存储格式**: PyTorch张量 (.pt文件)

### 5. 处理后的图数据
**用途**: 构建好的PyTorch Geometric图对象
**大小**: ~5-10GB

```bash
data/processed/
├── human_graph.pt               # ~2-3GB   人类PPI网络
├── mouse_graph.pt               # ~1-2GB   鼠PPI网络
├── yeast_graph.pt               # ~200-500MB 酵母PPI网络
├── human_node_mapping.json      # ~50MB    节点ID映射
├── mouse_node_mapping.json      # ~30MB
├── yeast_node_mapping.json      # ~10MB
└── preprocessing_stats.json     # ~1MB     预处理统计
```

### 6. 模型和日志数据
**用途**: 训练检查点和实验记录
**大小**: ~2-5GB

```bash
checkpoints/
├── hgcn_epoch_050.pt           # ~100-200MB 每个checkpoint
├── hgcn_epoch_100.pt
└── best_model.pt

logs/
├── train_eval.log              # ~50-100MB 训练日志
├── tensorboard/                # ~500MB    TensorBoard日志
└── wandb/                      # ~200MB    W&B缓存

results/
├── evaluation_results.json     # ~10MB     评估结果
├── ablation_study/             # ~100MB    消融实验
└── visualizations/             # ~200MB    可视化结果
```

## 🔄 数据获取流程

### 自动下载数据 (~8-10GB)
```bash
# 1. 下载STRING数据 (自动)
bash scripts/00_download_string.sh
# 下载时间: 1-3小时 (取决于网络)

# 2. UniProt ID映射 (自动)
python scripts/01_uniprot_id_mapping.py
# 处理时间: 30-60分钟
```

### 手动获取数据
**HuRI数据** (需手动下载):
- 访问: http://www.interactome-atlas.org/download
- 下载: HuRI Binary PPI dataset
- 放置: `data/raw/huri/`

### 特征提取 (~20-40GB)
```bash
# ESM-2特征提取 (最耗时)
bash scripts/03_extract_esm650m.sh
# 提取时间:
# - CPU: 10-20小时
# - GPU (16GB+): 3-6小时
```

## 💾 存储空间建议

### 最低配置
- **SSD**: 100GB (用于活跃数据和模型)
- **HDD**: 200GB (用于原始数据存储)

### 推荐配置
- **NVMe SSD**: 200GB (系统+活跃数据+特征)
- **SATA SSD/HDD**: 500GB (原始数据+备份)

### 理想配置
- **NVMe SSD**: 500GB+ (全部数据在高速存储)
- **网络存储**: 1TB+ (数据备份和归档)

## ⚡ 优化建议

### 1. 存储优化
```bash
# 使用压缩存储节省50%空间
tar -czf string_data_backup.tar.gz data/raw/string/
rm -rf data/raw/string/  # 原始文件

# 特征文件压缩
python -c "
import torch
features = torch.load('features/large_file.pt')
torch.save(features, 'features/large_file_compressed.pt', _use_new_zipfile_serialization=True)
"
```

### 2. 下载优化
```bash
# 并行下载加速
export DOWNLOAD_THREADS=4
bash scripts/00_download_string.sh --parallel
```

### 3. 内存映射(大数据集)
```python
# 使用内存映射减少RAM占用
dataset = torch.utils.data.TensorDataset(
    torch.from_file('features.dat', shared=True, size=total_size)
)
```

## 🚨 注意事项

### 网络要求
- **带宽**: ≥50Mbps (下载时间3-6小时)
- **流量**: ~10GB 初次下载
- **稳定性**: 支持断点续传

### 许可证
- **STRING**: 学术免费,商业需授权
- **HuRI**: CC BY 4.0许可
- **UniProt**: CC BY 4.0许可
- **ESM-2**: MIT许可

### 数据更新
- **STRING**: 年度更新 (建议半年检查)
- **HuRI**: 不定期更新
- **模型特征**: 一次提取长期使用

## 📈 数据统计

| 数据类型 | 压缩大小 | 解压大小 | 文件数量 | 更新频率 |
|---------|---------|---------|---------|---------|
| STRING原始 | ~8GB | ~25GB | ~36个 | 年度 |
| HuRI | ~150MB | ~150MB | 2个 | 不定期 |
| ID映射 | ~300MB | ~800MB | ~10个 | 季度 |
| ESM-2特征 | ~15GB | ~35GB | ~200个 | 一次性 |
| 处理图 | ~3GB | ~8GB | ~20个 | 重处理 |
| **总计** | **~26GB** | **~69GB** | **~268个** | - |

---

**建议**: 首次部署预留 **100GB** 存储空间,生产环境建议 **200GB+** 以支持多实验并行。
