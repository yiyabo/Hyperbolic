# PPI-HGCN 服务器部署建议

## 硬件配置推荐

### 最低配置 (开发/小规模实验)
- **CPU**: Intel Xeon 或 AMD EPYC，≥8核心
- **内存**: 32GB DDR4/DDR5
- **GPU**: NVIDIA RTX 3070/4060 Ti (8GB VRAM)
- **存储**: 500GB NVMe SSD
- **带宽**: ≥100Mbps

### 推荐配置 (生产/中等规模)
- **CPU**: Intel Xeon Gold 6xxx 系列或 AMD EPYC 7xxx 系列，≥16核心
- **内存**: 64GB DDR4/DDR5 ECC
- **GPU**: NVIDIA RTX 4090 (24GB) 或 A5000 (24GB)
- **存储**: 1TB NVMe SSD + 2TB HDD (数据存储)
- **带宽**: ≥1Gbps

### 高端配置 (大规模研究)
- **CPU**: Intel Xeon Platinum 或 AMD EPYC 9xxx，≥32核心
- **内存**: 128GB+ DDR5 ECC
- **GPU**: NVIDIA A100 (40GB/80GB) 或 H100 (80GB)
- **存储**: 2TB+ NVMe SSD RAID + 10TB+ 存储阵列
- **带宽**: ≥10Gbps

## 操作系统和环境设置

### 推荐操作系统
```bash
# Ubuntu 20.04/22.04 LTS (推荐)
# CentOS 8/RHEL 8
# Docker 支持: ✅
```

### CUDA 环境配置
```bash
# 推荐 CUDA 版本
CUDA 11.8 (推荐，兼容性最佳)
# 或者
CUDA 12.1 (最新特性支持)

# cuDNN 版本
cuDNN 8.x 对应 CUDA 版本

# 驱动版本
NVIDIA Driver ≥ 520.61 (for CUDA 11.8)
NVIDIA Driver ≥ 525.60 (for CUDA 12.1)
```

## 部署步骤

### 1. 系统依赖安装
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.9 python3.9-dev python3-pip
sudo apt install -y build-essential cmake git wget curl
sudo apt install -y libhdf5-dev libssl-dev

# CentOS/RHEL
sudo yum update -y
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python39 python39-devel python39-pip
sudo yum install -y cmake git wget curl hdf5-devel openssl-devel
```

### 2. NVIDIA 环境配置
```bash
# 安装 NVIDIA 驱动
sudo apt install nvidia-driver-520  # Ubuntu
# 或从官网下载对应版本

# 安装 CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 设置环境变量
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
nvidia-smi
nvcc --version
```

### 3. Python 环境配置
```bash
# 创建虚拟环境
python3.9 -m venv ppi_hgcn_env
source ppi_hgcn_env/bin/activate

# 升级 pip
pip install --upgrade pip setuptools wheel

# 安装 PyTorch (CUDA 11.8)
pip install torch==1.13.0+cu118 torchvision==0.14.0+cu118 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu118

# 安装 PyTorch Geometric
pip install torch-geometric==2.3.0
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu118.html
```

### 4. 项目部署
```bash
# 克隆项目
git clone <your-repo-url> ppi-hgcn
cd ppi-hgcn

# 安装项目依赖
pip install -r requirements.txt

# 验证安装
python scripts/test_installation.py

# 下载预训练模型和数据 (如果有)
bash scripts/00_download_string.sh
```

### 5. 性能优化配置
```bash
# 系统级优化
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf

# GPU 内存管理
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# OMP 线程数设置
export OMP_NUM_THREADS=8  # 根据 CPU 核心数调整
```

## 运行配置

### 1. 配置文件调整
```yaml
# cfg/server.yaml
device: 'cuda'
precision: 'float32'  # 服务器可用 float16 节省内存

# 批处理大小调整
train_batch_size: 1024    # 8GB VRAM
# train_batch_size: 2048  # 16GB VRAM
# train_batch_size: 4096  # 24GB+ VRAM

# 并行处理
num_workers: 8  # 根据 CPU 核心数调整
pin_memory: true
prefetch_factor: 2

# 内存优化
gradient_checkpointing: true  # 大模型开启
```

### 2. 监控和日志配置
```bash
# 安装监控工具
pip install wandb tensorboard psutil gpustat

# 启用 W&B (可选)
wandb login
export WANDB_PROJECT="ppi-hgcn-production"
export WANDB_MODE="online"

# GPU 监控脚本
cat > monitor_gpu.sh << 'EOF'
#!/bin/bash
while true; do
    echo "=== GPU Status $(date) ==="
    nvidia-smi
    echo "=== Memory Usage ==="
    free -h
    echo "========================"
    sleep 60
done
EOF
chmod +x monitor_gpu.sh
```

## 运行命令

### 完整训练流程
```bash
# 后台运行完整流程
nohup bash scripts/04_train_eval.sh \
    --config cfg/server.yaml \
    --wandb \
    --wandb-project ppi-hgcn-server > train.log 2>&1 &

# 查看进度
tail -f train.log

# 监控 GPU
./monitor_gpu.sh &
```

### 分步骤运行
```bash
# 1. 数据预处理 (可能需要几小时)
nohup python scripts/02_build_graphs.py --log-level INFO > preprocess.log 2>&1 &

# 2. 特征提取
nohup bash scripts/03_extract_esm650m.sh > feature_extract.log 2>&1 &

# 3. 模型训练
nohup python src/train/train_lp.py \
    --config cfg/server.yaml \
    --wandb \
    --log-level INFO > training.log 2>&1 &
```

## 故障排查

### 常见问题解决

**1. CUDA 内存不足**
```bash
# 减少批处理大小
export CUDA_VISIBLE_DEVICES=0
# 在配置中调整 batch_size

# 清理 GPU 内存
python -c "import torch; torch.cuda.empty_cache()"
```

**2. 依赖冲突**
```bash
# 重新创建环境
conda create -n ppi_clean python=3.9
conda activate ppi_clean
pip install -r requirements.txt
```

**3. 数据加载慢**
```bash
# 使用 SSD 存储数据
# 调整 num_workers 参数
# 启用 pin_memory
```

**4. 训练中断恢复**
```bash
# 使用 resume 功能
bash scripts/04_train_eval.sh --resume --config cfg/server.yaml
```

## 安全建议

### 网络安全
```bash
# 防火墙配置
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow from YOUR_IP_RANGE

# SSH 密钥认证
ssh-keygen -t rsa -b 4096
# 禁用密码登录
```

### 数据备份
```bash
# 自动备份脚本
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf /backup/ppi-hgcn-${DATE}.tar.gz \
    checkpoints/ results/ logs/ data/processed/
# 保留最新10个备份
ls -t /backup/ppi-hgcn-*.tar.gz | tail -n +11 | xargs rm -f
EOF

# 添加到 crontab
crontab -e
# 添加: 0 2 * * * /path/to/backup.sh
```

## 预期性能

### 训练时间估算 (基于硬件配置)

**最低配置 (RTX 3070)**
- 数据预处理: 4-6小时
- 特征提取: 2-3小时
- 模型训练: 12-24小时/epoch

**推荐配置 (RTX 4090)**
- 数据预处理: 2-3小时
- 特征提取: 1-1.5小时
- 模型训练: 6-12小时/epoch

**高端配置 (A100)**
- 数据预处理: 1-2小时
- 特征提取: 0.5-1小时
- 模型训练: 3-6小时/epoch

### 资源使用情况
- **峰值内存**: 24-48GB (取决于数据集大小)
- **GPU 内存**: 8-20GB (取决于批处理大小)
- **存储空间**: 100-500GB (包含原始数据和特征)
- **网络带宽**: 主要用于数据下载，训练期间需求较低

---

**注意**:
1. 首次运行会下载大量数据(STRING数据库约10GB)
2. ESM-2特征提取需要大量GPU内存
3. 建议在开始大规模训练前先用小数据集测试
4. 监控磁盘空间，日志和checkpoint会占用较多空间