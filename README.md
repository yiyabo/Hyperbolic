# 蛋白质相互作用预测：基于双曲几何的深度学习方法

## 项目概述

本项目基于双曲几何构建蛋白质相互作用（PPI）预测模型，采用 HGCN（双曲图卷积网络）在 Lorentz 流形上进行训练，并使用 ESM-2 650M 作为蛋白质序列特征提取器。项目旨在通过双曲空间的层次性质更好地建模蛋白质相互作用网络的复杂拓扑结构。

### 核心决策

- **训练主图**: HuRI（Human Reference Interactome）
- **外测数据**: STRING v12.0 物理子网络 + 跨物种测试（Mouse & Yeast）
- **特征提取**: ESM-2 650M mean-pooled 表征
- **模型架构**: HGCN (Lorentz) → 混合曲率（后续扩展）
- **数据拆分**: 冷蛋白 + 拓扑难负采样
- **时间拆分**: 暂不实施（后续可加入 BioGRID）

---

## 一、数据库选择与获取

### 1. 训练主图：HuRI（Human Reference Interactome）

#### 选择理由
- HuRI 是人类二元互作的参考图谱
- 采用酵母双分子荧光互补（Y2H）技术并辅以多种正交实验验证
- 数据为**物理二元互作**，质量较高，适合作为主训练图
- Nature 论文报告规模约 **53,000** 条高质量 PPI

#### 数据获取
- **官方门户**: [interactome-atlas.org](https://www.interactome-atlas.org/)
- **下载页面**: Download 页面提供 TSV 格式数据
- **ID 格式**: 通常为 Ensembl gene ID
- **备用访问**: 如门户访问不稳定，可通过 CCSB 页面的下载入口跳转

### 2. 外测数据：STRING v12.0（人类）

#### 选择理由
- STRING 覆盖 12,535 物种、59.3M 蛋白质、>200 亿条关联
- 支持**功能关联**与**物理子网络**两个子集
- 下载页支持按**物种过滤**和**置信度阈值**筛选（0.15/0.4/0.7/0.9）

#### 推荐文件
- **网络数据**: `protein.physical.links.v12.0.txt.gz`（物理子网）
- **置信度**: 建议使用 ≥0.7 或 ≥0.9 阈值
- **对齐文件**: 
  - `protein.info.v12.0.txt.gz`
  - `protein.aliases.v12.0.txt.gz`
- **序列数据**: `protein.sequences.v12.0.fa.gz`（如需序列）

### 3. 跨物种外测：Mouse（10090）& Yeast（4932）

#### 选择理由
- 两个物种注释最完善，跨物种迁移研究中常用
- STRING 下载页原生支持按 taxid 过滤并导出
- 适合批量评测不同物种间的模型泛化能力

#### 可选补充
- **BioGRID**: 可作为额外外测或未来时间拆分的数据源
- 官方下载仓库和格式说明完备

---

## 二、统一标识符与特征提取

### ID 统一化策略

#### 问题描述
- **HuRI**: 主要使用 Ensembl gene ID
- **STRING**: 使用 STRING protein ID

#### 解决方案
使用 **UniProt ID Mapping** 将各数据源的 ID 统一映射到 **UniProtKB accession**，然后进行合并和特征抽取。

#### 实施步骤
1. 使用 UniProt 官方提供的网页或 API 进行 ID 映射
2. 支持程序化批量访问
3. 建立统一的蛋白质索引系统

### 序列数据来源

#### 方案 A: UniProt 下载
- 从 UniProt 下载物种蛋白质 FASTA 文件
- 同时下载相应的映射表

#### 方案 B: STRING 序列包
- 直接使用 STRING 的 `protein.sequences.v12.0.fa.gz`
- 快速覆盖多物种需求

### ESM-2 650M 特征提取

#### 模型选择
- 使用 **facebookresearch/esm** 官方仓库
- 模型: `esm2_t33_650M_UR50D`
- 特征: **mean-pooled 表征**作为节点特征

#### 提取命令示例

```bash
python -m esm.extract \
  --model esm2_t33_650M_UR50D \
  --repr_layers 33 --include mean \
  --toks_per_batch 4096 \
  data/fasta/human.fasta \
  features/esm2_650m/human/
```

#### 多物种处理
- Mouse 和 Yeast 采用相同流程离线抽取并缓存
- 支持批量处理和断点续传

---

## 三、数据拆分与负采样策略

### 冷蛋白拆分（Inductive Setting）

#### 定义
测试集中出现的蛋白质不允许在训练边中出现，专注于评估模型对新蛋白质相互作用的外推能力。

#### 优势
- 更接近实际应用场景
- 避免数据泄露
- 测试模型的泛化能力

### 拓扑难负采样

#### 策略描述
采用**拓扑驱动的难负采样**，依据邻域、社团、路径结构挑选"难负样本"，而非简单的随机负采样。

#### 实施细节
- **负正比例**: 1:5
- **难负比例**: 50% 来自拓扑难负，50% 来自均匀负采样
- **理论支撑**: 近期生物信息学研究证明难负采样能显著提升模型泛化能力和评估公平性

#### 拓扑难负生成方法
- 同度数节点对
- 同社团内节点对
- 高邻域重叠但未连接的节点对
- 基于路径距离的负采样

---

## 四、项目结构

```
ppi-hgcn/
├─ cfg/
│  ├─ default.yaml                 # 全局配置
│  ├─ data_huri.yaml               # HuRI 数据配置
│  ├─ data_string_human.yaml       # STRING human 外测
│  ├─ data_string_mouse.yaml       # STRING mouse 外测
│  └─ data_string_yeast.yaml       # STRING yeast 外测
├─ data/
│  ├─ raw/
│  │  ├─ huri/                     # HuRI TSV 文件
│  │  ├─ string/                   # STRING 下载文件
│  │  └─ mapping/                  # UniProt 映射文件
│  └─ processed/
│     ├─ human_graph.pt           # 处理后的图数据
│     ├─ mouse_graph.pt
│     └─ yeast_graph.pt
├─ features/
│  └─ esm2_650m/
│     ├─ human/                    # ESM-2 特征文件
│     ├─ mouse/
│     └─ yeast/
├─ src/
│  ├─ dataio/
│  │  ├─ load_huri.py             # HuRI 数据加载
│  │  ├─ load_string.py           # STRING 数据加载
│  │  ├─ id_mapping_uniprot.py    # UniProt ID 映射
│  │  └─ neg_sampling.py          # 负采样策略
│  ├─ geometry/
│  │  ├─ manifolds.py             # 流形定义（Lorentz/Poincaré/Product）
│  │  └─ riemannian_optim.py      # 黎曼优化器封装
│  ├─ models/
│  │  ├─ hgcn.py                  # HGCN 主干模型
│  │  ├─ decoder.py               # 双曲距离解码器
│  │  └─ product_hgcn.py          # 混合曲率模型
│  ├─ train/
│  │  └─ train_lp.py              # 链路预测训练
│  ├─ eval/
│  │  ├─ metrics.py               # 评估指标
│  │  └─ evaluate_lp.py           # 评估流程
│  └─ utils/
│     ├─ logging.py               # 日志工具
│     └─ seed.py                  # 随机种子设置
├─ scripts/
│  ├─ 00_download_string.sh       # STRING 批量下载
│  ├─ 01_uniprot_id_mapping.py    # ID 映射脚本
│  ├─ 02_build_graphs.py          # 图数据构建
│  ├─ 03_extract_esm650m.sh       # ESM-2 特征提取
│  └─ 04_train_eval.sh            # 训练评估一体化
├─ README.md
└─ LICENSE
```

---

## 五、配置文件示例

### 全局配置（cfg/default.yaml）

```yaml
seed: 42
device: "cuda"
log_dir: "runs/"

data:
  train_graph: "data/processed/human_graph.pt"   # HuRI
  val_graph:   "data/processed/human_graph.pt"
  test_graphs:
    - "data/processed/human_graph.pt"            # 外测-人类（STRING）
    - "data/processed/mouse_graph.pt"            # 跨物种-鼠
    - "data/processed/yeast_graph.pt"            # 跨物种-酵母
  split:
    mode: "cold_protein"                         # 冷蛋白拆分
    ratio: [0.8, 0.1, 0.1]
  negatives:
    ratio: 5
    hard_frac: 0.5
    strategy: "topology_driven"                  # 或 "uniform"

features:
  kind: "esm2_650m_mean"
  dir:  "features/esm2_650m/"

model:
  backbone: "hgcn_lorentz"
  dim: 128
  layers: 3
  curvature: "learnable"                         # 每层可学曲率
  dropout: 0.1
  decoder:
    type: "hyperbolic_distance"
    tau_learnable: true

optim:
  name: "riemannian_adam"
  lr: 1.5e-3
  weight_decay: 0.0
  epochs: 100
  early_stop_patience: 10
  batch_size: 8192

eval:
  metrics: ["aupr", "auroc", "hits@10", "calibration"]
  threshold_opt: "Youden"
  report_per_graph: true
```

---

## 六、模型架构

### HGCN（双曲图卷积网络）

#### 核心思想
- 在 **Lorentz 流形**上执行图卷积操作
- 利用双曲空间的负曲率特性建模层次结构
- 每层学习自适应曲率参数

#### 实现要点
- 参考 HazyResearch/hgcn 的 PyTorch 实现
- 支持链路预测任务管线
- 减少自实现错误

### 双曲距离解码器

#### 功能
- 计算双曲空间中节点对的距离
- 可学习温度参数
- 输出相互作用概率

#### 数学基础
在 Lorentz 流形上，两点间的双曲距离为：
```
d_L(x, y) = arccosh(-⟨x, y⟩_L)
```
其中 ⟨·,·⟩_L 是 Lorentz 内积。

---

## 七、训练流程

### 训练主循环伪代码

```python
# 初始化
manifold = geoopt.manifolds.Lorentz()
model = HGCN(dim=cfg.model.dim, layers=cfg.model.layers, 
              manifold=manifold, learn_curvature=True)
decoder = HyperbolicDistDecoder(tau_learnable=cfg.model.decoder.tau_learnable, 
                                manifold=manifold)

optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=cfg.optim.lr)
sampler = NegativeSampler(mode=cfg.data.negatives.strategy, 
                          ratio=cfg.data.negatives.ratio,
                          hard_frac=cfg.data.negatives.hard_frac)

# 训练循环
for epoch in range(cfg.optim.epochs):
    for pos_edges in train_loader:
        # 负采样
        neg_edges = sampler.sample(graph, pos_edges)
        
        # 前向传播
        logits = decoder(model, pos_edges, neg_edges)
        loss = bce_with_logits(logits.labels, logits.scores)
        
        # 反向传播（黎曼优化）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证
    val_report = evaluate(model, decoder, val_loader)
    if early_stop(val_report): 
        break

# 测试评估
for test_graph in test_graphs:
    test_report = evaluate(model, decoder, test_graph)
```

### 关键实现细节

#### 黎曼优化
- 使用 **geoopt** 库的 RiemannianAdam 优化器
- 确保参数更新保持在流形上
- 支持各种双曲流形（Lorentz、Poincaré等）

#### 批处理策略
- 边级别的批处理
- 动态负采样
- 内存效率优化

---

## 八、数据处理脚本

### HuRI 数据加载（load_huri.py）

#### 功能
- 读取 HuRI TSV 文件
- 去除自环
- 处理为无向边
- Ensembl gene ID → UniProtKB accession 映射

#### 关键步骤
```python
def load_huri(filepath, id_mapping):
    # 读取 TSV
    edges = pd.read_csv(filepath, sep='\t')
    
    # 去自环
    edges = edges[edges['protein1'] != edges['protein2']]
    
    # ID 映射
    edges['protein1'] = edges['protein1'].map(id_mapping)
    edges['protein2'] = edges['protein2'].map(id_mapping)
    
    # 构建图
    graph = build_graph(edges)
    return graph
```

### STRING 数据加载（load_string.py）

#### 功能
- 读取 STRING 物理链接文件
- 按 taxid 过滤物种
- 按 combined_score 阈值过滤
- STRING protein ID → UniProtKB 映射

#### 参数设置
- **置信度阈值**: ≥0.7 或 ≥0.9
- **物种过滤**: Human(9606), Mouse(10090), Yeast(4932)
- **链接类型**: 物理相互作用子网络

### UniProt ID 映射（id_mapping_uniprot.py）

#### API 调用
- 封装 UniProt Retrieve/ID Mapping API
- 支持批量处理
- 断点续传和重试机制

#### 实现示例
```python
def map_ids_batch(source_ids, from_db, to_db):
    url = "https://rest.uniprot.org/idmapping/run"
    data = {
        'from': from_db,
        'to': to_db,
        'ids': ','.join(source_ids)
    }
    response = requests.post(url, data=data)
    return parse_mapping_result(response)
```

### 负采样实现（neg_sampling.py）

#### 拓扑驱动采样
```python
class TopologyDrivenSampler:
    def __init__(self, graph, hard_frac=0.5):
        self.graph = graph
        self.hard_frac = hard_frac
        
    def sample_hard_negatives(self, pos_edges, num_samples):
        # 基于度数相似性
        degree_similar = self.sample_by_degree_similarity(pos_edges, num_samples//3)
        
        # 基于共同邻居
        common_neighbor = self.sample_by_common_neighbors(pos_edges, num_samples//3)
        
        # 基于社团结构  
        community_based = self.sample_by_community(pos_edges, num_samples//3)
        
        return degree_similar + common_neighbor + community_based
```

---

## 九、评估指标与基线

### 评估指标

#### 主要指标
- **AUPR（Area Under Precision-Recall Curve）**: 主要指标，对不平衡数据更敏感
- **AUROC（Area Under ROC Curve）**: 传统分类指标
- **Hits@K**: 排序质量评估，K=10,50,100
- **校准曲线**: 置信度-召回率分析

#### 指标计算
```python
def evaluate_link_prediction(model, decoder, test_data):
    predictions = model.predict(test_data.edges)
    labels = test_data.labels
    
    # 计算各指标
    aupr = average_precision_score(labels, predictions)
    auroc = roc_auc_score(labels, predictions)
    hits_at_k = compute_hits_at_k(predictions, labels, k=[10, 50, 100])
    calibration = compute_calibration_curve(predictions, labels)
    
    return {
        'aupr': aupr,
        'auroc': auroc, 
        'hits@k': hits_at_k,
        'calibration': calibration
    }
```

### 基线模型

#### 第一批基线
- **GCN（欧几里得）**: 使用相同的 ESM-2 特征
- **GAT（欧几里得）**: 注意力机制对照

#### 后续基线
- **Product-HGCN**: 混合曲率模型
- **传统方法**: 基于拓扑特征的方法

### 评估集设计

#### 人类数据
- **训练**: HuRI 冷蛋白拆分
- **测试**: STRING 物理子网络（≥0.7, ≥0.9）

#### 跨物种数据
- **Mouse**: STRING Mouse 数据
- **Yeast**: STRING Yeast 数据
- **评估**: 零样本迁移能力

---

## 十、扩展路线图

### 短期扩展（3-6个月）

#### 1. 混合曲率 Product Manifold
- **背景**: 在 δ-hyperbolicity 明显的图上常有进一步提升
- **实现**: `product_hgcn.py` 模块，开关式配置
- **评估**: 与单一曲率模型对比

#### 2. HGAT（双曲注意力）
- **动机**: STRING 包含多种证据类型，适合注意力机制
- **应用**: 异质图神经网络
- **优势**: 更好的可解释性

### 中期扩展（6-12个月）

#### 3. 时间拆分验证
- **数据源**: BioGRID 月度更新数据
- **拆分策略**: "train-past / test-future"
- **目标**: 防止时间泄漏，评估时间泛化

#### 4. GO 双曲先验
- **工具**: HiG2Vec（GO 与基因在 Poincaré 球面嵌入）
- **集成**: 作为正则化项或特征拼接
- **数据**: 开源权重（Human/Mouse/Yeast）

### 长期扩展（1年+）

#### 5. STRING 跨物种对齐
- **资源**: STRING v12 跨物种嵌入和序列嵌入 HDF5 文件
- **方法**: 对齐学习或知识蒸馏
- **应用**: 提升跨物种预测准确性

#### 6. δ-hyperbolicity 诊断
- **目的**: 量化证据支持双曲/混合曲率选择
- **实现**: δ 近似评估模块
- **应用**: 自适应几何选择

---

## 十一、实验配置

### 第一版默认配置

#### 训练设置
- **数据**: HuRI 冷蛋白拆分（8:1:1）
- **特征**: ESM-2 650M mean-pooled
- **模型**: HGCN（Lorentz，3层，128维）
- **曲率**: 每层可学习
- **负采样**: 1:5 比例，50% 拓扑难负

#### 评估设置
- **外测**: 
  - STRING-Human 物理子网（≥0.7 和 ≥0.9）
  - STRING-Mouse（同阈值）
  - STRING-Yeast（同阈值）
- **指标**: AUPR（主要）、AUROC、Hits@K、校准
- **对照**: GCN（欧几里得，相同特征）

#### 技术设置
- **优化器**: RiemannianAdam
- **学习率**: 1.5e-3
- **批大小**: 8192
- **早停**: 验证集 10 轮无改善
- **日志**: WandB 或 TSV（可配置）

---

## 十二、快速开始

### 环境要求

```bash
# Python 版本
python >= 3.8

# 核心依赖
torch >= 1.9.0
torch-geometric >= 2.0.0
geoopt >= 0.3.0
fair-esm >= 0.4.0

# 数据处理
pandas >= 1.3.0
numpy >= 1.20.0
scipy >= 1.7.0
networkx >= 2.6.0

# 评估与可视化
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

### 快速启动流程

```bash
# 1. 克隆并进入项目目录
git clone <repository-url>
cd ppi-hgcn

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载数据
bash scripts/00_download_string.sh

# 4. ID 映射
python scripts/01_uniprot_id_mapping.py

# 5. 构建图数据
python scripts/02_build_graphs.py

# 6. 提取 ESM-2 特征
bash scripts/03_extract_esm650m.sh

# 7. 训练和评估
bash scripts/04_train_eval.sh
```

### 配置自定义

修改 `cfg/default.yaml` 中的参数：
- 调整模型维度和层数
- 更改负采样策略
- 设置不同的置信度阈值
- 选择评估指标

---

## 参考文献与资源

### 数据源
- **HuRI**: [Interactome Atlas](https://www.interactome-atlas.org/)
- **STRING v12**: [STRING Database](https://string-db.org/)
- **UniProt**: [ID Mapping Service](https://www.uniprot.org/id-mapping)

### 模型与方法
- **ESM-2**: [facebookresearch/esm](https://github.com/facebookresearch/esm)
- **HGCN**: [HazyResearch/hgcn](https://github.com/HazyResearch/hgcn)
- **geoopt**: [Riemannian Optimization](https://github.com/geoopt/geoopt)

### 理论背景
- Chami et al. "Hyperbolic Graph Convolutional Neural Networks" (NeurIPS 2019)
- Nickel & Kiela "Poincaré Embeddings for Learning Hierarchical Representations" (NeurIPS 2017)
- 拓扑难负采样相关研究
- HiG2Vec: GO 双曲嵌入方法

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 贡献指南

欢迎提交 Issue 和 Pull Request。请确保：

1. 代码符合 PEP 8 规范
2. 包含适当的测试用例
3. 更新相关文档
4. 遵循现有的项目结构

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至项目维护者
- 参与项目讨论

---

*最后更新: 2024年12月*
