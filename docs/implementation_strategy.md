# HGCN (双曲图卷积网络) 核心实现策略文档

**项目**: PPI-HGCN - 基于双曲几何的蛋白质相互作用预测  
**文档版本**: v1.0  
**创建日期**: 2024年12月  
**目标**: 完整实现HGCN模型，支持消融实验和科研验证

---

## 1. 数学理论基础

### 1.1 双曲几何基础理论

#### 1.1.1 Lorentz 模型 (Hyperboloid Model)

**定义**: Lorentz 模型定义在 $\mathbb{R}^{d+1}$ 空间中的双曲面上：

$$\mathbb{H}^d = \{x \in \mathbb{R}^{d+1} : \langle x, x \rangle_L = -1, x_0 > 0\}$$

其中 Lorentz 内积定义为：
$$\langle x, y \rangle_L = -x_0 y_0 + \sum_{i=1}^d x_i y_i$$

**切空间**: 在点 $x \in \mathbb{H}^d$ 处的切空间为：
$$T_x \mathbb{H}^d = \{v \in \mathbb{R}^{d+1} : \langle v, x \rangle_L = 0\}$$

**黎曼度量**: 切空间上的度量由 Lorentz 内积的限制给出：
$$g_x(u, v) = \langle u, v \rangle_L \quad \text{for } u, v \in T_x \mathbb{H}^d$$

#### 1.1.2 双曲距离

两点 $x, y \in \mathbb{H}^d$ 之间的双曲距离：
$$d_{\mathbb{H}}(x, y) = \text{arccosh}(-\langle x, y \rangle_L)$$

**曲率 $c$ 下的距离**: 在半径 $R=1/\sqrt{c}$ 的双曲面上：
$$\boxed{d_c(x, y) = \frac{1}{\sqrt{c}} \cdot \text{arccosh}(-c \langle x, y \rangle_L)}$$

**数值稳定性**: 当 $-c\langle x, y \rangle_L$ 接近 1 时，使用稳定公式：
$$d_c(x, y) = \frac{1}{\sqrt{c}} \log(-c\langle x, y \rangle_L + \sqrt{(-c\langle x, y \rangle_L)^2 - 1})$$

#### 1.1.3 指数映射和对数映射

**指数映射** $\exp_x^{(c)}: T_x \mathbb{H}^d \to \mathbb{H}^d$:
$$\boxed{\exp_x^{(c)}(v) = \cosh(\sqrt{c}\|v\|_L) x + \frac{1}{\sqrt{c}} \sinh(\sqrt{c}\|v\|_L) \frac{v}{\|v\|_L}}$$

其中 $\|v\|_L = \sqrt{\langle v, v \rangle_L}$

**对数映射** $\log_x^{(c)}: \mathbb{H}^d \to T_x \mathbb{H}^d$:
$$\boxed{\log_x^{(c)}(y) = \frac{\text{arccosh}(-c\langle x, y \rangle_L)}{\sqrt{(-c\langle x, y \rangle_L)^2 - 1}} (y + c\langle x, y \rangle_L x)}$$

**小步长近似**: 当 $\sqrt{c}\|v\|_L < 10^{-3}$ 时：
$$\exp_x^{(c)}(v) \approx \left(1 + \frac{c\|v\|_L^2}{2}\right)x + \left(1 + \frac{c\|v\|_L^2}{6}\right)v$$

#### 1.1.4 平行传输

从 $x$ 到 $y$ 的平行传输 (半径 $R=1/\sqrt{c}$ 版本):
$$\boxed{P_{x \to y}^{(c)}(v) = v - \frac{\langle y, v \rangle_L}{1 - c\langle x, y \rangle_L} (x + y)}$$

**数值稳定性**: 分母需要 `clamp_min(1e-15)` 防止接近测地线端点时不稳定。

**正交化修正**: 返回前执行一次投影消除数值漂移：
$$v \leftarrow v - \frac{\langle v, y \rangle_L}{\langle y, y \rangle_L} y$$

### 1.2 可学习曲率

引入可学习曲率参数 $c \in \mathbb{R}^+$，缩放双曲空间：

**缩放双曲空间**: $\mathbb{H}^d_c = \{x \in \mathbb{R}^{d+1} : \langle x, x \rangle_L = -1/c, x_0 > 0\}$

**参数化策略**: $c = \text{softplus}(\tilde{c}) + c_{\min}$，其中 $c_{\min} = 10^{-4}$

**曲率粒度**: 默认**逐层曲率**$c^{(l)}$，配置`model.curvature_per_layer: true|false`

**数学意义**: $c \to 0$ 时趋近欧几里得空间，$c$ 越大双曲性越强

**超曲面归一化**: 每层末尾执行归一化防止漂移：
$$x \leftarrow \frac{x}{\sqrt{-c\langle x, x \rangle_L}} \quad \text{若}x_0 \leq 0\text{则}x \leftarrow -x$$

---

## 2. HGCN 模型理论

### 2.1 双曲图卷积层

#### 2.1.0 双曲输入层

**欧几里得到双曲的首层映射**: 节点初始特征为欧几里得（ESM-2），需要首层映射到双曲空间：

$$x^{(0)} = \exp_o^{(c)}(W_{\text{in}} \phi(\text{ESM}(i)) + b_{\text{in}})$$

其中 $\phi$ 可包含 LayerNorm/Dropout/线性降维，$W_{\text{in}}: \mathbb{R}^{d_{\text{esm}}} \to \mathbb{R}^d$

**目的**: 确保第1层HGCN的输入已在$\mathbb{H}_c$上

#### 2.1.1 特征变换

**线性变换**: 在切空间中执行
$$h_i^{(l)} = W^{(l)} x_i^{(l)} + b^{(l)}$$

其中 $W^{(l)} \in \mathbb{R}^{d^{(l+1)} \times d^{(l)}}$, $b^{(l)} \in \mathbb{R}^{d^{(l+1)}}$

#### 2.1.2 邻域聚合

**聚合函数**: 在Lorentz流形上的加权聚合
$$\tilde{h}_i^{(l)} = \text{LorentzAgg}(\{h_j^{(l)} : j \in \mathcal{N}(i) \cup \{i\}\})$$

**Lorentz加权聚合**: 使用Minkowski加权和后归一化
$$\boxed{\bar{x} = \frac{\sum_j \alpha_{ij} x_j}{\sqrt{-\langle \sum_j \alpha_{ij} x_j, \sum_j \alpha_{ij} x_j \rangle_L}} \cdot \frac{1}{\sqrt{c}}}$$

其中 $\alpha_{ij}$ 是归一化权重（$\sum_j \alpha_{ij} = 1, \alpha_{ij} \geq 0$）。

**时样性检查与兜底**: 若加权和不是时样向量（$-\langle s,s \rangle_L \leq 0$），兜底为切空间均值：
$$\bar{x} = \exp_{b}^{(c)}\left(\sum_j \alpha_{ij} \log_{b}^{(c)}(x_j)\right)$$

其中基点$b$可配置：`origin`使用$o=(1/\sqrt{c},0,\ldots,0)$，`nodewise`使用当前节点。

**上片保证**: 聚合后若$x_0 \leq 0$，整体乘以$-1$回到上片（top sheet）确保时向一致性。

#### 2.1.3 非线性激活

**双曲激活函数**: 保持双曲约束的激活
$$\sigma_{\mathbb{H}}(x) = \exp_o^{(c)}(\sigma(\log_o^{(c)}(x)))$$

其中 $o = (1/\sqrt{c}, 0, ..., 0)$ 是基点，$\sigma$ 是欧几里得激活函数

**基点选择**: 统一使用 $o = (R, 0, \ldots, 0), R = 1/\sqrt{c}$ 作为基点

### 2.2 HGCN 层的完整前向传播

$$X^{(l+1)} = \sigma_{\mathbb{H}}(\text{LorentzAgg}(\text{HypLin}(X^{(l)})))$$

详细步骤：
1. **切空间投影**: $X_{\text{tan}} = \log_o^{(c)}(X^{(l)})$
2. **线性变换**: $H = X_{\text{tan}} W^{(l)} + b^{(l)}$
3. **双曲投影**: $X_{\text{hyp}} = \exp_o^{(c)}(H)$
4. **邻域聚合**: $\tilde{X} = \text{LorentzAgg}(X_{\text{hyp}}, A)$
5. **激活函数**: $X^{(l+1)} = \sigma_{\mathbb{H}}(\tilde{X})$
6. **超曲面归一化**: $X^{(l+1)} \leftarrow \frac{X^{(l+1)}}{\sqrt{-\langle X^{(l+1)}, X^{(l+1)} \rangle_L}} \cdot \frac{1}{\sqrt{c}}$

---

## 3. 链路预测解码器

### 3.1 双曲距离解码器

**距离计算**:
$$s_{ij} = -d_c(h_i, h_j) = -\frac{1}{\sqrt{c}} \cdot \text{arccosh}(-c \langle h_i, h_j \rangle_L)$$

**温度缩放**:
$$p_{ij} = \sigma(s_{ij} / \tau)$$

其中 $\tau$ 是可学习温度参数

**数值稳定性**: `arccosh` 的自变量需要 `clamp_min(1 + 1e-7)`

**权重归一化**: 确保聚合权重$\sum_j \alpha_{ij} = 1$，推荐使用row-softmax注意力或度归一化

### 3.2 双线性解码器 (消融对比)

**欧几里得双线性**（在切空间执行）:
$$s_{ij} = \log_o^{(c)}(h_i)^T R \log_o^{(c)}(h_j)$$

其中 $R$ 是关系矩阵，避免直接在Lorentz坐标上做双线性操作

---

## 4. 高度模块化架构设计

### 4.1 核心模块划分

#### 4.1.1 几何模块 (`geometry/`)

```
geometry/
├── base_manifold.py          # 抽象流形基类
├── lorentz_manifold.py       # Lorentz 流形实现
├── poincare_manifold.py      # Poincaré 流形实现 (消融对比)
├── euclidean_manifold.py     # 欧几里得空间 (基线对比)
├── product_manifold.py       # 乘积流形 (混合曲率)
└── ops.py                    # 几何操作函数
```

#### 4.1.2 模型组件 (`models/components/`)

```
models/components/
├── hyperbolic_layers.py      # 双曲层基类
├── hgcn_layer.py            # HGCN 卷积层
├── linear_layer.py          # 双曲线性层
├── aggregators.py           # 聚合函数集合
├── activations.py           # 双曲激活函数
├── attention.py             # 双曲注意力机制
└── dropout.py               # 双曲 Dropout
```

#### 4.1.3 解码器模块 (`models/decoders/`)

```
models/decoders/
├── base_decoder.py          # 解码器基类
├── distance_decoder.py      # 双曲距离解码器
├── bilinear_decoder.py      # 双线性解码器
├── dot_product_decoder.py   # 点积解码器
└── attention_decoder.py     # 注意力解码器
```

#### 4.1.4 主模型类 (`models/`)

```
models/
├── base_model.py           # 模型基类
├── hgcn.py                 # 主HGCN模型
├── gcn_baseline.py         # 欧几里得GCN基线
├── gat_baseline.py         # 欧几里得GAT基线
└── mixed_curvature.py      # 混合曲率模型
```

### 4.2 消融实验支持

#### 4.2.1 可配置组件

```yaml
# 消融实验配置示例
ablation:
  geometry:
    type: "lorentz"           # lorentz, poincare, euclidean
    learnable_curvature: true # true, false
    
  aggregation:
    type: "mean"              # mean, attention, max, sum
    hyperbolic: true          # true (双曲聚合), false (欧几里得聚合)
    basepoint: "origin"       # origin, nodewise
    
  decoder:
    type: "distance"          # distance, bilinear, dot_product
    temperature: "learnable"  # learnable, fixed, none
    
  activation:
    type: "relu"              # relu, tanh, elu
    hyperbolic: true          # true (双曲激活), false (欧几里得激活)
    
  residual:
    in_tangent: true          # true (切空间残差), false (无残差)
    beta: 0.5                 # 残差权重: X^(l+1) = exp_o(log_o(X̃) + β·log_o(X^(l)))
    
model:
  curvature_per_layer: true   # true (逐层曲率), false (全局曲率)
```

#### 4.2.2 组件工厂模式

```python
class ComponentFactory:
    @staticmethod
    def create_manifold(config):
        """根据配置创建流形"""
        
    @staticmethod
    def create_layer(config):
        """根据配置创建卷积层"""
        
    @staticmethod
    def create_decoder(config):
        """根据配置创建解码器"""

#### 4.2.3 统一API接口设计

**几何模块统一接口**:

```python
# geometry/lorentz_manifold.py
class Lorentz:
    """Lorentz流形统一接口"""
    def __init__(self, c: float): 
        self.c = c
        
    def dot(self, x, y): 
        """Lorentz内积: -x₀y₀ + Σxᵢyᵢ"""
        
    def norm_tan(self, v): 
        """切向量的Lorentz范数: √⟨v,v⟩_L"""
        
    def proj(self, x): 
        """投影回双曲面: ⟨x,x⟩_L=-1/c且x₀>0"""
        
    def exp(self, x, v): 
        """指数映射: T_x H → H"""
        
    def log(self, x, y): 
        """对数映射: H → T_x H"""
        
    def dist(self, x, y): 
        """双曲距离计算"""
        
    def transport(self, x, y, v): 
        """平行传输: T_x H → T_y H"""
```

**聚合器统一接口**:

```python
# models/components/aggregators.py
class LorentzAggregator:
    """Lorentz聚合器统一接口"""
    def __init__(self, manifold, mode="mean"):
        self.M = manifold  # Lorentz流形对象
        self.mode = mode   # mean, attention, max, sum
        
    def forward(self, X, edge_index, alpha=None):
        """
        Args:
            X: (N, d+1) 节点特征在双曲空间
            edge_index: (2, E) 边索引
            alpha: (E,) 可选权重，自动row-softmax归一化
        Returns:
            (N, d+1) 聚合结果，满足⟨x,x⟩_L = -1/c
        """
        # 确保时样性，含兜底机制
        # 返回严格满足双曲约束的结果 (x₀>0, ⟨x,x⟩_L=-1/c)
```

---

## 5. 数学正确性验证方案

### 5.1 单元测试 - 几何操作

#### 5.1.1 流形约束测试
```python
def test_hyperboloid_constraint():
    """验证点在双曲面上: <x,x>_L = -1/c"""
    x = sample_hyperbolic_point(dim=5, curvature=1.0)
    constraint = lorentz_inner_product(x, x)
    assert abs(constraint + 1.0) < 1e-6

def test_tangent_space_orthogonality():
    """验证切向量与基点正交: <v,x>_L = 0"""
    x = sample_hyperbolic_point(dim=5, curvature=1.0)
    v = sample_tangent_vector(x)
    orthogonality = lorentz_inner_product(v, x)
    assert abs(orthogonality) < 1e-6
```

#### 5.1.2 距离计算测试
```python
def test_hyperbolic_distance_properties():
    """验证双曲距离的数学性质"""
    x, y, z = sample_hyperbolic_points(3, dim=5, curvature=1.0)
    
    # 对称性: d(x,y) = d(y,x)
    assert abs(hyperbolic_distance(x, y) - hyperbolic_distance(y, x)) < 1e-6
    
    # 非负性: d(x,y) >= 0
    assert hyperbolic_distance(x, y) >= 0
    
    # 三角不等式: d(x,z) <= d(x,y) + d(y,z)
    assert hyperbolic_distance(x, z) <= hyperbolic_distance(x, y) + hyperbolic_distance(y, z) + 1e-6
```

#### 5.1.3 指数/对数映射测试
```python
def test_exp_log_inverse():
    """验证指数和对数映射互为逆运算"""
    x = sample_hyperbolic_point(dim=5, curvature=1.0)
    v = sample_tangent_vector(x)
    
    # exp_x(log_x(y)) = y
    y = exponential_map(x, v)
    v_recovered = logarithmic_map(x, y)
    assert torch.allclose(v, v_recovered, atol=1e-5)

def test_isometric_invariance():
    """验证Lorentz等距变换下的不变性"""
    x, y = sample_hyperbolic_points(2, dim=5, curvature=1.0)
    A = sample_lorentz_transformation(dim=5)  # 洛伦兹群变换
    
    # 距离不变性: d(Ax, Ay) = d(x, y)
    dist_orig = hyperbolic_distance(x, y)
    dist_transformed = hyperbolic_distance(A @ x, A @ y)
    assert abs(dist_orig - dist_transformed) < 1e-6
    
    # 内积不变性: <Ax, Ay>_L = <x, y>_L
    inner_orig = lorentz_inner_product(x, y)
    inner_transformed = lorentz_inner_product(A @ x, A @ y)
    assert abs(inner_orig - inner_transformed) < 1e-6
```

### 5.2 集成测试 - 模型组件

#### 5.2.1 梯度计算验证
```python
def test_gradient_correctness():
    """数值梯度 vs 自动微分梯度"""
    model = SimpleHGCN(input_dim=32, hidden_dim=16)
    x, edge_index = create_test_graph()
    
    # 自动微分梯度
    output = model(x, edge_index)
    loss = output.sum()
    loss.backward()
    auto_grad = model.layers[0].weight.grad.clone()
    
    # 数值梯度
    numerical_grad = compute_numerical_gradient(model, x, edge_index)
    
    assert torch.allclose(auto_grad, numerical_grad, atol=1e-4)
```

#### 5.2.2 前向传播数值稳定性
```python
def test_forward_stability():
    """测试极端输入下的数值稳定性"""
    model = HGCN(input_dim=32, hidden_dim=16, curvature=1e-6)
    
    # 测试各种极端情况
    test_cases = [
        create_large_norm_input(),      # 大范数输入
        create_small_norm_input(),      # 小范数输入
        create_sparse_graph(),          # 稀疏图
        create_dense_graph(),           # 稠密图
    ]
    
    for x, edge_index in test_cases:
        output = model(x, edge_index)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
```

### 5.3 端到端验证

#### 5.3.1 小数据集过拟合测试
```python
def test_overfit_small_dataset():
    """在小数据集上测试过拟合能力"""
    # 创建简单的可学习数据集
    train_data = create_simple_ppi_graph(num_nodes=20, num_edges=50)
    
    model = HGCN(input_dim=16, hidden_dim=32, num_layers=2)
    optimizer = RiemannianAdam(model.parameters(), lr=0.01)
    
    # 训练至过拟合
    for epoch in range(1000):
        loss = train_epoch(model, train_data, optimizer)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    # 验证最终性能
    final_auc = evaluate_model(model, train_data)
    assert final_auc > 0.95, f"Failed to overfit: AUC = {final_auc}"
```

#### 5.3.2 聚合闭性测试
```python
def test_aggregation_closure():
    """测试聚合后的点仍在双曲面上"""
    neighbors = sample_hyperbolic_points(5, dim=5, curvature=1.0)
    weights = torch.softmax(torch.randn(5), dim=0)  # 归一化权重
    
    # Lorentz加权聚合
    aggregated = lorentz_barycenter(neighbors, weights, curvature=1.0)
    
    # 验证双曲约束
    constraint = lorentz_inner_product(aggregated, aggregated)
    expected = -1.0  # <x,x>_L = -1/c, c=1.0
    assert abs(constraint - expected) < 1e-5

def test_exp_log_roundtrip_drift():
    """监控exp/log往返漂移率"""
    x = sample_hyperbolic_point(dim=5, curvature=1.0)
    v = sample_tangent_vector(x)
    
    # 往返测试: v -> exp -> log -> v'
    y = exponential_map(x, v)
    v_recovered = logarithmic_map(x, y)
    
    # 计算漂移率
    drift_ratio = torch.norm(v_recovered - v) / torch.norm(v)
    
    # 记录P50/P95分位数用于监控
    assert drift_ratio < 1e-5, f"High drift ratio: {drift_ratio}"

def test_parallel_transport_properties():
    """验证平行传输的保范性和正交性"""
    x, y = sample_hyperbolic_points(2, dim=5, curvature=1.0)
    v = sample_tangent_vector(x)                      # <v,x>_L = 0
    v2 = transport(x, y, v)
    
    # 正交性保持: <v2,y>_L = 0
    assert abs(lorentz_dot(v2, y)) < 1e-6
    
    # 范数保持: <v,v>_L = <v2,v2>_L  
    assert torch.allclose(lorentz_dot(v, v), lorentz_dot(v2, v2), atol=1e-6)

def test_projection_top_sheet():
    """验证投影到上片的正确性"""
    x = random_ambient_vector()                       # 可能 x0 <= 0
    x_proj = manifold.proj(x)                         # 应落到 <x,x>=-1/c 且 x0>0
    
    # 双曲约束
    constraint = lorentz_dot(x_proj, x_proj)
    assert abs(constraint + 1.0/c) < 1e-6
    
    # 上片约束
    assert (x_proj[...,0] > 0).all()

def test_performance_vs_baselines():
    """与欧几里得基线对比"""
    dataset = load_test_dataset()
    
    # HGCN 模型
    hgcn = HGCN(manifold="lorentz")
    hgcn_results = train_and_evaluate(hgcn, dataset)
    
    # 欧几里得 GCN 基线
    gcn = GCN(manifold="euclidean")
    gcn_results = train_and_evaluate(gcn, dataset)
    
    print(f"HGCN AUC: {hgcn_results['auc']:.4f}")
    print(f"GCN AUC: {gcn_results['auc']:.4f}")
    
    # HGCN 应该优于 GCN (在具有层次结构的图上)
    assert hgcn_results['auc'] >= gcn_results['auc'] - 0.01
```

### 5.4 数学一致性验证

#### 5.4.1 曲率参数收敛性
```python
def test_curvature_convergence():
    """测试可学习曲率的合理收敛"""
    model = HGCN(learnable_curvature=True, initial_curvature=1.0)
    
    # 记录训练过程中曲率变化
    curvature_history = []
    for epoch in range(100):
        train_epoch(model, dataset)
        curvature_history.append(model.curvature.item())
    
    # 验证曲率收敛到合理范围
    final_curvature = curvature_history[-1]
    assert 0.01 <= final_curvature <= 10.0
    
    # 验证收敛稳定性
    recent_std = np.std(curvature_history[-10:])
    assert recent_std < 0.1
```

---

## 6. 分步实现计划

### 阶段 1: 几何基础 (1周)

**目标**: 实现稳定可靠的双曲几何操作

**任务**:
1. `lorentz_manifold.py` - Lorentz 流形类
2. 基础几何操作: 距离、指数/对数映射、平行传输
3. 数值稳定性优化
4. 完整单元测试套件

**验收标准**:
- 通过所有几何操作单元测试
- 数值稳定性测试通过
- 与现有geoopt实现对比验证

### 阶段 2: 模型组件 (1.5周)

**目标**: 实现模块化的HGCN组件

**任务**:
1. `hgcn_layer.py` - 核心HGCN卷积层
2. `aggregators.py` - 多种聚合函数
3. `activations.py` - 双曲激活函数
4. `distance_decoder.py` - 双曲距离解码器

**验收标准**:
- 每个组件独立测试通过
- 梯度计算正确性验证
- 模块间接口兼容性测试

### 阶段 3: 完整模型 (1周)

**目标**: 集成完整的HGCN模型

**任务**:
1. `hgcn.py` - 主模型类
2. 多层堆叠和残差连接
3. 可学习曲率集成
4. 模型保存/加载

**验收标准**:
- 前向传播正确性测试
- 小数据集过拟合测试
- 模型序列化测试

### 阶段 4: 训练框架 (1周)

**目标**: 实现完整的训练和评估流程

**任务**:
1. `train_lp.py` - 链路预测训练循环
2. 黎曼优化器集成
3. 早停和学习率调度
4. `evaluate_lp.py` - 评估指标计算

**验收标准**:
- 训练流程端到端测试
- 评估指标正确性验证
- 与现有结果对比验证

### 阶段 5: 消融实验支持 (0.5周)

**目标**: 完善消融实验基础设施

**任务**:
1. 配置驱动的组件工厂
2. 基线模型实现 (GCN, GAT)
3. 实验脚本和结果分析
4. 性能分析工具

**验收标准**:
- 消融实验配置测试
- 基线对比实验
- 性能分析报告

---

## 7. 关键实现细节

### 7.1 数值稳定性策略

#### 7.1.1 双曲距离计算
```python
def acosh_stable(z: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """数值稳定的 arccosh 计算"""
    z = torch.clamp(z, min=1.0 + eps)
    return torch.acosh(z)

def lorentz_distance_c(x, y, c: float, eps: float = 1e-7):
    """曲率c下的Lorentz距离"""
    # x,y: (..., d+1), <x,x>_L = -1/c
    mprod = lorentz_dot(x, y)               # -x0*y0 + sum_{i>=1} x_i*y_i
    z = -c * mprod                          # argument to acosh
    return (1.0 / math.sqrt(c)) * acosh_stable(z, eps)
```

#### 7.1.2 指数映射稳定化
```python
def exp_map_c(x, v, c: float, eps: float = 1e-9):
    """曲率c下的稳定指数映射"""
    v_norm = torch.clamp(torch.sqrt(lorentz_dot(v, v)), min=eps)  # tangent norm
    sqrt_c_vnorm = torch.sqrt(c) * v_norm
    
    # 小范数情况使用泰勒展开 (阈值: sqrt(c)*||v||_L < SMALL_GEODESIC)
    SMALL_GEODESIC = 1e-3  # 统一小步长阈值，可跨模块管理
    small_norm_mask = sqrt_c_vnorm < SMALL_GEODESIC
    result = torch.zeros_like(x)
    
    # 小范数: exp_x(v) ≈ (1 + c|v|²/2)x + (1 + c|v|²/6)v
    small_result = ((1 + c * v_norm**2 / 2).unsqueeze(-1) * x + 
                   (1 + c * v_norm**2 / 6).unsqueeze(-1) * v)
    result[small_norm_mask] = small_result[small_norm_mask]
    
    # 正常情况
    normal_mask = ~small_norm_mask
    coef1 = torch.cosh(sqrt_c_vnorm[normal_mask])
    coef2 = torch.sinh(sqrt_c_vnorm[normal_mask]) / (torch.sqrt(c) * v_norm[normal_mask])
    result[normal_mask] = (coef1.unsqueeze(-1) * x[normal_mask] + 
                          coef2.unsqueeze(-1) * v[normal_mask])
    
    return project_to_hyperboloid(result, c)  # 归一化到 <y,y>_L = -1/c

def lorentz_barycenter(xs, weights, c: float, eps: float = 1e-9):
    """Lorentz加权聚合含时样性检查"""
    # xs: (k, d+1) 邻居；weights: (k,) 已归一化且非负
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert (weights >= 0).all()
    
    y = torch.sum(weights.view(-1,1) * xs, dim=0)
    
    # 时样性检查
    lorentz_norm_sq = -lorentz_dot(y, y)
    if lorentz_norm_sq <= 0:
        # 兜底: 切空间均值
        o = torch.zeros_like(xs[0])
        o[0] = 1.0 / math.sqrt(c)  # 基点
        tangent_vecs = [log_map_c(o, x, c) for x in xs]
        avg_tangent = torch.sum(weights.view(-1,1) * torch.stack(tangent_vecs), dim=0)
        return exp_map_c(o, avg_tangent, c)
    
    # 正常归一化
    denom = torch.clamp(torch.sqrt(lorentz_norm_sq), min=eps)
    result = (1.0 / (denom * math.sqrt(c))) * y
    
    # 确保在上片 (x₀ > 0)
    if result[0] <= 0:
        result = -result
    
    return result
```

### 7.2 维度约定与接口对齐

#### 7.2.1 切空间张量形状统一
- **切空间**: 统一使用`(N, d)`形状，丢弃第0维
- **Ambient空间**: 统一使用`(N, d+1)`形状
- **YAML中的dim**: 指切空间维度d，ambient自动为d+1

```python
def to_tangent_dplus1(vec_d):
    """切空间向量 R^d -> ambient形式 (0, vec_d)"""
    return torch.cat([torch.zeros_like(vec_d[...,:1]), vec_d], dim=-1)
    
def from_tangent_dplus1(vec_dplus1):
    """ambient形式切向量 -> 切空间向量 R^d"""
    return vec_dplus1[..., 1:]
```

### 7.3 性能优化策略

#### 7.3.1 批处理优化
- 向量化所有几何操作
- 使用 `torch.bmm` 进行批量矩阵运算
- 避免显式循环

#### 7.3.2 内存优化
- 梯度检查点用于大模型
- 就地操作 (`inplace=True`) 当安全时
- 动态图切换到静态图

### 7.4 调试和监控

#### 7.4.1 数值健康检查
```python
def check_hyperbolic_health(x, name="tensor"):
    """检查双曲张量的数值健康性"""
    # 检查 NaN 和 Inf
    assert not torch.isnan(x).any(), f"{name} contains NaN"
    assert not torch.isinf(x).any(), f"{name} contains Inf"
    
    # 检查双曲约束
    constraint = lorentz_inner_product(x, x)
    expected = -torch.ones_like(constraint)
    error = torch.abs(constraint - expected)
    assert error.max() < 1e-4, f"{name} violates hyperboloid constraint"
```

#### 7.4.2 训练监控
- 曲率参数变化曲线
- 梯度范数监控  
- 双曲约束违反统计
- exp/log往返漂移率分位数（P50/P95）
- 径向分布监控: $\rho_i = \text{arccosh}(-c\langle x_i, o \rangle_L)$
- 违反超曲约束的样本占比（应≈0）
- x₀≤0被翻转的比例（应极低）
- 逐层曲率参数$c^{(l)}$变化曲线
- 数值稳定性指标

---

## 8. 预期挑战和解决方案

### 8.1 数值挑战

**挑战**: 双曲几何操作的数值不稳定性
**解决方案**: 
- 多精度计算支持
- 自适应数值稳定性策略  
- 广泛的边界情况测试

### 8.2 性能挑战

**挑战**: 双曲操作比欧几里得操作慢
**解决方案**:
- 高度优化的CUDA实现
- 批处理和向量化
- 关键路径性能分析

### 8.3 调试挑战

**挑战**: 双曲空间的可视化和调试困难
**解决方案**:
- 降维投影可视化工具
- 丰富的数值健康检查
- 分层调试策略

---

## 9. 成功指标

### 9.1 技术指标

- ✅ 所有单元测试通过率 > 99%
- ✅ 数值稳定性测试通过率 100%
- ✅ 小数据集过拟合能力 (AUC > 0.95)
- ✅ 与基线对比性能提升 > 5%

### 9.2 科研指标

- ✅ 数学实现完全符合理论推导
- ✅ 支持完整的消融实验
- ✅ 代码可重现已发表结果
- ✅ 高质量的文档和注释

### 9.3 工程指标

- ✅ 模块化程度高，易于扩展
- ✅ 配置驱动，参数调节简单
- ✅ 性能可接受 (< 2x 欧几里得基线)
- ✅ 内存使用合理

---

## 10. 风险评估和缓解

### 10.1 高风险项

**风险**: Lorentz 流形实现错误
**概率**: 中等  
**影响**: 致命
**缓解**: 与多个参考实现对比，详细单元测试

**风险**: 数值不稳定导致训练失败
**概率**: 高
**影响**: 严重
**缓解**: 多层数值稳定性保护，自适应精度

### 10.2 中风险项

**风险**: 性能不佳影响实验效率
**概率**: 中等
**影响**: 中等  
**缓解**: 分阶段性能优化，必要时使用C++扩展

**风险**: 黎曼优化器集成问题
**概率**: 低
**影响**: 中等
**缓解**: 紧密跟随geoopt文档，简单测试案例验证

---

## 结论

本实现策略以**数学严谨性**和**高度模块化**为核心原则，确保：

1. **理论完整性** - 所有关键数学组件都有完整实现
2. **实验灵活性** - 支持广泛的消融实验和对比研究  
3. **工程质量** - 高质量代码，易于维护和扩展
4. **科研可用** - 满足严格的科研标准和重现性要求

通过分阶段实现和全面测试验证，我们将构建一个可靠、高效、易用的HGCN实现框架。