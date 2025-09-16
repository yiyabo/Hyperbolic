#!/bin/bash

# 训练和评估脚本
# 一体化脚本，包含数据预处理、模型训练、评估的完整流程

set -e  # 遇到错误立即退出

# 配置
CONFIG_FILE="cfg/default.yaml"
LOG_FILE="logs/train_eval.log"
CHECKPOINT_DIR="checkpoints"
RESULTS_DIR="results"
WANDB_PROJECT="ppi-hgcn"

# 默认参数
SKIP_PREPROCESSING=false
SKIP_TRAINING=false
SKIP_EVALUATION=false
RESUME_TRAINING=false
DEBUG_MODE=false
USE_WANDB=false

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config|-c)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --skip-preprocessing)
                SKIP_PREPROCESSING=true
                shift
                ;;
            --skip-training)
                SKIP_TRAINING=true
                shift
                ;;
            --skip-evaluation)
                SKIP_EVALUATION=true
                shift
                ;;
            --resume|-r)
                RESUME_TRAINING=true
                shift
                ;;
            --debug|-d)
                DEBUG_MODE=true
                shift
                ;;
            --wandb|-w)
                USE_WANDB=true
                shift
                ;;
            --wandb-project)
                WANDB_PROJECT="$2"
                USE_WANDB=true
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  --config, -c FILE          Use specified config file (default: cfg/default.yaml)
  --skip-preprocessing       Skip data preprocessing steps
  --skip-training           Skip model training
  --skip-evaluation         Skip model evaluation
  --resume, -r              Resume training from last checkpoint
  --debug, -d               Run in debug mode (verbose logging)
  --wandb, -w               Use Weights & Biases for logging
  --wandb-project PROJECT   Specify W&B project name
  --help, -h                Show this help message

Examples:
  $0                                    # Run complete pipeline
  $0 --config cfg/custom.yaml          # Use custom config
  $0 --skip-preprocessing --resume     # Resume training without preprocessing
  $0 --wandb --wandb-project my-ppi    # Use W&B logging
EOF
}

# 设置日志
setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")"

    if [[ "$DEBUG_MODE" == "true" ]]; then
        export LOG_LEVEL="DEBUG"
    else
        export LOG_LEVEL="INFO"
    fi

    echo "$(date): Starting PPI-HGCN training pipeline..." | tee "$LOG_FILE"
    echo "Config file: $CONFIG_FILE" | tee -a "$LOG_FILE"
    echo "Log level: $LOG_LEVEL" | tee -a "$LOG_FILE"
    echo "Skip preprocessing: $SKIP_PREPROCESSING" | tee -a "$LOG_FILE"
    echo "Skip training: $SKIP_TRAINING" | tee -a "$LOG_FILE"
    echo "Skip evaluation: $SKIP_EVALUATION" | tee -a "$LOG_FILE"
    echo "Resume training: $RESUME_TRAINING" | tee -a "$LOG_FILE"
    echo "Use W&B: $USE_WANDB" | tee -a "$LOG_FILE"
}

# 检查环境
check_environment() {
    echo "$(date): Checking environment..." | tee -a "$LOG_FILE"

    # 检查Python
    if ! command -v python &> /dev/null; then
        echo "Error: Python not found" | tee -a "$LOG_FILE"
        exit 1
    fi

    # 检查必需的Python包
    local required_packages=("torch" "torch_geometric" "geoopt" "yaml" "pandas" "numpy" "sklearn" "networkx")

    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" &> /dev/null; then
            echo "Error: Python package '$package' not installed" | tee -a "$LOG_FILE"
            exit 1
        fi
    done

    # 检查CUDA可用性
    if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep "True" &> /dev/null; then
        echo "CUDA is available" | tee -a "$LOG_FILE"
        export DEVICE="cuda"
    else
        echo "CUDA not available, using CPU" | tee -a "$LOG_FILE"
        export DEVICE="cpu"
    fi

    # 检查配置文件
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "Error: Config file not found: $CONFIG_FILE" | tee -a "$LOG_FILE"
        exit 1
    fi

    # 检查W&B
    if [[ "$USE_WANDB" == "true" ]]; then
        if ! python -c "import wandb" &> /dev/null; then
            echo "Error: wandb package not installed but --wandb specified" | tee -a "$LOG_FILE"
            exit 1
        fi
        export WANDB_PROJECT="$WANDB_PROJECT"
        export WANDB_MODE="online"
    else
        export WANDB_MODE="disabled"
    fi

    echo "Environment check passed" | tee -a "$LOG_FILE"
}

# 数据预处理
run_preprocessing() {
    if [[ "$SKIP_PREPROCESSING" == "true" ]]; then
        echo "$(date): Skipping preprocessing as requested" | tee -a "$LOG_FILE"
        return 0
    fi

    echo "$(date): Starting data preprocessing..." | tee -a "$LOG_FILE"

    # 1. 下载STRING数据（如果需要）
    if [[ ! -f "data/raw/string/.download_complete" ]]; then
        echo "Downloading STRING data..." | tee -a "$LOG_FILE"
        if ! bash scripts/00_download_string.sh 2>&1 | tee -a "$LOG_FILE"; then
            echo "Error: STRING data download failed" | tee -a "$LOG_FILE"
            return 1
        fi
    else
        echo "STRING data already downloaded" | tee -a "$LOG_FILE"
    fi

    # 2. ID映射（如果需要）
    if [[ ! -f "data/raw/mapping/.mapping_complete" ]]; then
        echo "Creating ID mappings..." | tee -a "$LOG_FILE"
        if ! python scripts/01_uniprot_id_mapping.py --log-level "$LOG_LEVEL" 2>&1 | tee -a "$LOG_FILE"; then
            echo "Error: ID mapping failed" | tee -a "$LOG_FILE"
            return 1
        fi
    else
        echo "ID mappings already created" | tee -a "$LOG_FILE"
    fi

    # 3. 构建图数据（如果需要）
    if [[ ! -f "data/processed/.build_complete" ]]; then
        echo "Building graph data..." | tee -a "$LOG_FILE"
        if ! python scripts/02_build_graphs.py --log-level "$LOG_LEVEL" 2>&1 | tee -a "$LOG_FILE"; then
            echo "Error: Graph building failed" | tee -a "$LOG_FILE"
            return 1
        fi
    else
        echo "Graph data already built" | tee -a "$LOG_FILE"
    fi

    # 4. 提取ESM-2特征（如果需要）
    if [[ ! -f "features/esm2_650m/.extraction_complete" ]]; then
        echo "Extracting ESM-2 features..." | tee -a "$LOG_FILE"
        if ! bash scripts/03_extract_esm650m.sh 2>&1 | tee -a "$LOG_FILE"; then
            echo "Error: Feature extraction failed" | tee -a "$LOG_FILE"
            return 1
        fi
    else
        echo "ESM-2 features already extracted" | tee -a "$LOG_FILE"
    fi

    echo "$(date): Preprocessing completed successfully" | tee -a "$LOG_FILE"
    return 0
}

# 模型训练
run_training() {
    if [[ "$SKIP_TRAINING" == "true" ]]; then
        echo "$(date): Skipping training as requested" | tee -a "$LOG_FILE"
        return 0
    fi

    echo "$(date): Starting model training..." | tee -a "$LOG_FILE"

    # 创建checkpoint目录
    mkdir -p "$CHECKPOINT_DIR"

    # 构建训练命令
    local train_cmd="python src/train/train_lp.py --config $CONFIG_FILE --log-level $LOG_LEVEL"

    if [[ "$RESUME_TRAINING" == "true" ]]; then
        train_cmd="$train_cmd --resume"
    fi

    if [[ "$USE_WANDB" == "true" ]]; then
        train_cmd="$train_cmd --wandb --wandb-project $WANDB_PROJECT"
    fi

    # 运行训练
    echo "Running training command: $train_cmd" | tee -a "$LOG_FILE"

    if eval "$train_cmd" 2>&1 | tee -a "$LOG_FILE"; then
        echo "$(date): Training completed successfully" | tee -a "$LOG_FILE"
        return 0
    else
        echo "$(date): Training failed" | tee -a "$LOG_FILE"
        return 1
    fi
}

# 模型评估
run_evaluation() {
    if [[ "$SKIP_EVALUATION" == "true" ]]; then
        echo "$(date): Skipping evaluation as requested" | tee -a "$LOG_FILE"
        return 0
    fi

    echo "$(date): Starting model evaluation..." | tee -a "$LOG_FILE"

    # 创建results目录
    mkdir -p "$RESULTS_DIR"

    # 查找最新的checkpoint
    local latest_checkpoint=$(find "$CHECKPOINT_DIR" -name "*.pt" -type f -exec ls -t {} + | head -n 1)

    if [[ -z "$latest_checkpoint" ]]; then
        echo "Error: No checkpoint found for evaluation" | tee -a "$LOG_FILE"
        return 1
    fi

    echo "Using checkpoint: $latest_checkpoint" | tee -a "$LOG_FILE"

    # 构建评估命令
    local eval_cmd="python src/eval/evaluate_lp.py --config $CONFIG_FILE --checkpoint $latest_checkpoint --output-dir $RESULTS_DIR --log-level $LOG_LEVEL"

    # 运行评估
    echo "Running evaluation command: $eval_cmd" | tee -a "$LOG_FILE"

    if eval "$eval_cmd" 2>&1 | tee -a "$LOG_FILE"; then
        echo "$(date): Evaluation completed successfully" | tee -a "$LOG_FILE"

        # 显示结果摘要
        if [[ -f "$RESULTS_DIR/evaluation_summary.json" ]]; then
            echo "Evaluation Summary:" | tee -a "$LOG_FILE"
            python -c "
import json
with open('$RESULTS_DIR/evaluation_summary.json', 'r') as f:
    results = json.load(f)
for dataset, metrics in results.items():
    print(f'  {dataset}:')
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f'    {metric}: {value:.4f}')
        else:
            print(f'    {metric}: {value}')
" | tee -a "$LOG_FILE"
        fi

        return 0
    else
        echo "$(date): Evaluation failed" | tee -a "$LOG_FILE"
        return 1
    fi
}

# 清理函数
cleanup() {
    echo "$(date): Pipeline interrupted, cleaning up..." | tee -a "$LOG_FILE"

    # 终止子进程
    if [[ -n "$training_pid" ]]; then
        kill "$training_pid" 2>/dev/null || true
    fi

    exit 1
}

# 主函数
main() {
    # 设置信号处理
    trap cleanup SIGINT SIGTERM

    # 解析命令行参数
    parse_arguments "$@"

    # 设置日志
    setup_logging

    # 检查环境
    check_environment

    # 记录开始时间
    local start_time=$(date +%s)

    # 运行预处理
    if ! run_preprocessing; then
        echo "$(date): Pipeline failed during preprocessing" | tee -a "$LOG_FILE"
        exit 1
    fi

    # 运行训练
    if ! run_training; then
        echo "$(date): Pipeline failed during training" | tee -a "$LOG_FILE"
        exit 1
    fi

    # 运行评估
    if ! run_evaluation; then
        echo "$(date): Pipeline failed during evaluation" | tee -a "$LOG_FILE"
        exit 1
    fi

    # 计算总耗时
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))

    echo "$(date): Pipeline completed successfully!" | tee -a "$LOG_FILE"
    echo "Total runtime: ${hours}h ${minutes}m ${seconds}s" | tee -a "$LOG_FILE"

    # 创建完成标记
    touch "${RESULTS_DIR}/.pipeline_complete"

    # 生成最终报告
    generate_final_report

    exit 0
}

# 生成最终报告
generate_final_report() {
    local report_file="${RESULTS_DIR}/pipeline_report.md"

    cat > "$report_file" << EOF
# PPI-HGCN Pipeline Report

**Generated on:** $(date)
**Config file:** $CONFIG_FILE
**Total runtime:** ${hours}h ${minutes}m ${seconds}s

## Pipeline Status

- ✅ Data preprocessing completed
- ✅ Model training completed
- ✅ Model evaluation completed

## Configuration Summary

\`\`\`yaml
$(head -20 "$CONFIG_FILE")
...
\`\`\`

## Results Files

- Training logs: $LOG_FILE
- Model checkpoints: $CHECKPOINT_DIR/
- Evaluation results: $RESULTS_DIR/

## Next Steps

1. Review evaluation metrics in \`$RESULTS_DIR/evaluation_summary.json\`
2. Analyze training curves in logs or W&B dashboard
3. Compare results with baseline models
4. Consider hyperparameter tuning if needed

EOF

    echo "Final report generated: $report_file" | tee -a "$LOG_FILE"
}

# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
