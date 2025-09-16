#!/bin/bash

# ESM-2 650M 特征提取脚本
# 批量提取Human, Mouse, Yeast蛋白质的ESM-2 650M特征

set -e  # 遇到错误立即退出

# 配置
ESM_MODEL="esm2_t33_650M_UR50D"
FEATURE_DIR="features/esm2_650m"
LOG_FILE="logs/extract_esm650m.log"
BATCH_SIZE=4096
REPR_LAYERS=33

# 物种配置
declare -A SPECIES
SPECIES[human]=9606
SPECIES[mouse]=10090
SPECIES[yeast]=4932

# Python环境检查
check_python_environment() {
    echo "$(date): Checking Python environment..." | tee -a "$LOG_FILE"

    if ! command -v python &> /dev/null; then
        echo "Error: Python not found" | tee -a "$LOG_FILE"
        exit 1
    fi

    # 检查ESM是否安装
    if ! python -c "import esm" &> /dev/null; then
        echo "Error: ESM package not installed. Please install with: pip install fair-esm" | tee -a "$LOG_FILE"
        exit 1
    fi

    # 检查torch是否安装
    if ! python -c "import torch" &> /dev/null; then
        echo "Error: PyTorch not installed" | tee -a "$LOG_FILE"
        exit 1
    fi

    # 检查GPU可用性
    if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep "True" &> /dev/null; then
        echo "CUDA is available - will use GPU acceleration" | tee -a "$LOG_FILE"
        export CUDA_VISIBLE_DEVICES=0
    else
        echo "CUDA not available - will use CPU (slower)" | tee -a "$LOG_FILE"
    fi

    echo "Python environment check passed" | tee -a "$LOG_FILE"
}

# 创建必要目录
create_directories() {
    echo "$(date): Creating directories..." | tee -a "$LOG_FILE"

    mkdir -p "$FEATURE_DIR"
    mkdir -p "logs"
    mkdir -p "data/fasta"

    for species in "${!SPECIES[@]}"; do
        mkdir -p "${FEATURE_DIR}/${species}"
    done

    echo "Directories created successfully" | tee -a "$LOG_FILE"
}

# 准备FASTA文件
prepare_fasta_files() {
    echo "$(date): Preparing FASTA files..." | tee -a "$LOG_FILE"

    for species in "${!SPECIES[@]}"; do
        local taxid=${SPECIES[$species]}
        local fasta_file="data/fasta/${species}.fasta"
        local string_fasta="data/raw/string/${taxid}.protein.sequences.v12.0.fa.gz"

        echo "Processing ${species} FASTA..." | tee -a "$LOG_FILE"

        # 检查是否已有FASTA文件
        if [[ -f "$fasta_file" && -s "$fasta_file" ]]; then
            echo "  FASTA file already exists: $fasta_file" | tee -a "$LOG_FILE"
            continue
        fi

        # 从STRING序列文件提取
        if [[ -f "$string_fasta" ]]; then
            echo "  Extracting from STRING sequences: $string_fasta" | tee -a "$LOG_FILE"

            # 解压并处理FASTA文件
            zcat "$string_fasta" | python3 -c "
import sys
import re

# 读取并清理FASTA文件
count = 0
current_header = None
current_seq = []

for line in sys.stdin:
    line = line.strip()
    if line.startswith('>'):
        # 输出前一个序列
        if current_header and current_seq:
            seq = ''.join(current_seq)
            # 过滤掉太短的序列
            if len(seq) >= 20:
                print(current_header)
                print(seq)
                count += 1

        # 开始新序列
        current_header = line
        current_seq = []
    else:
        # 移除非氨基酸字符
        clean_seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', line.upper())
        if clean_seq:
            current_seq.append(clean_seq)

# 输出最后一个序列
if current_header and current_seq:
    seq = ''.join(current_seq)
    if len(seq) >= 20:
        print(current_header)
        print(seq)
        count += 1

print(f'# Processed {count} sequences', file=sys.stderr)
" > "$fasta_file"

            # 检查生成的文件
            if [[ -f "$fasta_file" && -s "$fasta_file" ]]; then
                local seq_count=$(grep -c "^>" "$fasta_file")
                echo "  Successfully created FASTA with $seq_count sequences" | tee -a "$LOG_FILE"
            else
                echo "  Warning: Failed to create FASTA file for $species" | tee -a "$LOG_FILE"
            fi
        else
            echo "  Warning: STRING sequences file not found: $string_fasta" | tee -a "$LOG_FILE"
        fi
    done

    echo "FASTA preparation completed" | tee -a "$LOG_FILE"
}

# 提取ESM特征
extract_esm_features() {
    local species=$1
    local fasta_file="data/fasta/${species}.fasta"
    local output_dir="${FEATURE_DIR}/${species}"

    echo "$(date): Extracting ESM-2 features for ${species}..." | tee -a "$LOG_FILE"

    # 检查输入文件
    if [[ ! -f "$fasta_file" ]]; then
        echo "Error: FASTA file not found: $fasta_file" | tee -a "$LOG_FILE"
        return 1
    fi

    # 检查是否已经提取过
    if [[ -f "${output_dir}/.extraction_complete" ]]; then
        echo "Features already extracted for ${species}, skipping..." | tee -a "$LOG_FILE"
        return 0
    fi

    # 计算序列数量
    local seq_count=$(grep -c "^>" "$fasta_file")
    echo "Processing $seq_count sequences for ${species}" | tee -a "$LOG_FILE"

    # 运行ESM提取
    echo "Starting feature extraction..." | tee -a "$LOG_FILE"

    # 使用临时输出目录避免部分完成的提取
    local temp_output_dir="${output_dir}_temp"
    mkdir -p "$temp_output_dir"

    # ESM提取命令
    local extract_cmd="python -m esm.extract \
        --model $ESM_MODEL \
        --repr_layers $REPR_LAYERS \
        --include mean \
        --toks_per_batch $BATCH_SIZE \
        '$fasta_file' \
        '$temp_output_dir'"

    echo "Running command: $extract_cmd" | tee -a "$LOG_FILE"

    # 执行提取（带超时和重试）
    local max_retries=2
    local retry_count=0
    local success=false

    while [[ $retry_count -lt $max_retries && "$success" == "false" ]]; do
        echo "Extraction attempt $((retry_count + 1))/$max_retries" | tee -a "$LOG_FILE"

        if timeout 7200 bash -c "$extract_cmd" 2>&1 | tee -a "$LOG_FILE"; then
            success=true
            echo "Feature extraction completed successfully" | tee -a "$LOG_FILE"
        else
            echo "Extraction attempt $((retry_count + 1)) failed" | tee -a "$LOG_FILE"
            retry_count=$((retry_count + 1))

            if [[ $retry_count -lt $max_retries ]]; then
                echo "Waiting 30 seconds before retry..." | tee -a "$LOG_FILE"
                sleep 30
            fi
        fi
    done

    if [[ "$success" == "true" ]]; then
        # 移动临时目录到最终位置
        if [[ -d "$output_dir" ]]; then
            rm -rf "$output_dir"
        fi
        mv "$temp_output_dir" "$output_dir"

        # 验证输出
        local feature_files=$(find "$output_dir" -name "*.pt" | wc -l)
        echo "Generated $feature_files feature files" | tee -a "$LOG_FILE"

        if [[ $feature_files -gt 0 ]]; then
            # 创建完成标记
            touch "${output_dir}/.extraction_complete"

            # 创建特征统计信息
            python3 -c "
import torch
import os
from pathlib import Path
import json

feature_dir = Path('$output_dir')
pt_files = list(feature_dir.glob('*.pt'))

stats = {
    'species': '$species',
    'model': '$ESM_MODEL',
    'num_proteins': len(pt_files),
    'feature_files': []
}

# 检查几个文件获取维度信息
for i, pt_file in enumerate(pt_files[:5]):
    try:
        data = torch.load(pt_file, map_location='cpu')
        if 'mean_representations' in data and $REPR_LAYERS in data['mean_representations']:
            features = data['mean_representations'][$REPR_LAYERS]
            stats['feature_dim'] = features.shape[-1]
            break
    except Exception as e:
        print(f'Warning: Could not load {pt_file}: {e}')

# 保存统计信息
with open(feature_dir / 'feature_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f'Feature extraction completed for $species:')
print(f'  Files: {len(pt_files)}')
if 'feature_dim' in stats:
    print(f'  Dimension: {stats[\"feature_dim\"]}')
" | tee -a "$LOG_FILE"

            echo "Feature extraction for ${species} completed successfully" | tee -a "$LOG_FILE"
            return 0
        else
            echo "Error: No feature files generated for ${species}" | tee -a "$LOG_FILE"
            return 1
        fi
    else
        # 清理临时目录
        rm -rf "$temp_output_dir"
        echo "Error: Feature extraction failed for ${species} after $max_retries attempts" | tee -a "$LOG_FILE"
        return 1
    fi
}

# 主提取流程
extract_all_features() {
    echo "$(date): Starting ESM-2 feature extraction for all species..." | tee -a "$LOG_FILE"

    local total_species=${#SPECIES[@]}
    local successful_extractions=0
    local failed_extractions=0

    for species in "${!SPECIES[@]}"; do
        echo "" | tee -a "$LOG_FILE"
        echo "$(date): Processing $species ($(($successful_extractions + $failed_extractions + 1))/$total_species)" | tee -a "$LOG_FILE"
        echo "================================================" | tee -a "$LOG_FILE"

        if extract_esm_features "$species"; then
            successful_extractions=$((successful_extractions + 1))
            echo "✓ $species completed successfully" | tee -a "$LOG_FILE"
        else
            failed_extractions=$((failed_extractions + 1))
            echo "✗ $species failed" | tee -a "$LOG_FILE"
        fi
    done

    # 提取摘要
    echo "" | tee -a "$LOG_FILE"
    echo "$(date): Feature Extraction Summary" | tee -a "$LOG_FILE"
    echo "================================================" | tee -a "$LOG_FILE"
    echo "Total species: $total_species" | tee -a "$LOG_FILE"
    echo "Successful extractions: $successful_extractions" | tee -a "$LOG_FILE"
    echo "Failed extractions: $failed_extractions" | tee -a "$LOG_FILE"

    # 显示特征目录大小
    if command -v du &> /dev/null; then
        local total_size=$(du -sh "$FEATURE_DIR" | cut -f1)
        echo "Total feature size: $total_size" | tee -a "$LOG_FILE"
    fi

    # 创建全局完成标记
    if [[ $failed_extractions -eq 0 ]]; then
        touch "${FEATURE_DIR}/.extraction_complete"
        echo "$(date): All feature extractions completed successfully!" | tee -a "$LOG_FILE"

        # 创建特征README
        cat > "${FEATURE_DIR}/README.md" << EOF
# ESM-2 650M Features

Extracted on: $(date)
Script: 03_extract_esm650m.sh
Model: $ESM_MODEL

## Species Features

### Human (Homo sapiens)
- Directory: features/esm2_650m/human/
- Model layers: $REPR_LAYERS (mean-pooled)
- Batch size: $BATCH_SIZE

### Mouse (Mus musculus)
- Directory: features/esm2_650m/mouse/
- Model layers: $REPR_LAYERS (mean-pooled)
- Batch size: $BATCH_SIZE

### Yeast (Saccharomyces cerevisiae)
- Directory: features/esm2_650m/yeast/
- Model layers: $REPR_LAYERS (mean-pooled)
- Batch size: $BATCH_SIZE

## Usage

Each protein has a corresponding .pt file containing:
- mean_representations: Dict with layer representations
- Layer $REPR_LAYERS contains the 1280-dim mean-pooled features

Load features in Python:
\`\`\`python
import torch
data = torch.load('protein_id.pt')
features = data['mean_representations'][$REPR_LAYERS]  # Shape: [1280]
\`\`\`

## Notes

- Features are mean-pooled across sequence length
- Model: ESM-2 650M (esm2_t33_650M_UR50D)
- Feature dimension: 1280
- All sequences >= 20 amino acids
EOF

        return 0
    else
        echo "$(date): Some feature extractions failed. Please check the log and retry." | tee -a "$LOG_FILE"
        return 1
    fi
}

# 验证提取的特征
validate_features() {
    echo "$(date): Validating extracted features..." | tee -a "$LOG_FILE"

    python3 -c "
import torch
from pathlib import Path
import json

feature_dir = Path('$FEATURE_DIR')
all_valid = True
validation_results = {}

for species in ['human', 'mouse', 'yeast']:
    species_dir = feature_dir / species
    validation_results[species] = {
        'exists': species_dir.exists(),
        'feature_files': 0,
        'valid_files': 0,
        'feature_dim': None,
        'errors': []
    }

    if not species_dir.exists():
        validation_results[species]['errors'].append('Directory does not exist')
        all_valid = False
        continue

    pt_files = list(species_dir.glob('*.pt'))
    validation_results[species]['feature_files'] = len(pt_files)

    if len(pt_files) == 0:
        validation_results[species]['errors'].append('No feature files found')
        all_valid = False
        continue

    # 验证前几个文件
    valid_count = 0
    for pt_file in pt_files[:min(10, len(pt_files))]:
        try:
            data = torch.load(pt_file, map_location='cpu')
            if 'mean_representations' in data and $REPR_LAYERS in data['mean_representations']:
                features = data['mean_representations'][$REPR_LAYERS]
                if validation_results[species]['feature_dim'] is None:
                    validation_results[species]['feature_dim'] = features.shape[-1]
                if features.shape[-1] == 1280:  # ESM-2 650M dimension
                    valid_count += 1
                else:
                    validation_results[species]['errors'].append(f'Wrong feature dimension: {features.shape[-1]}')
            else:
                validation_results[species]['errors'].append(f'Missing expected keys in {pt_file.name}')
        except Exception as e:
            validation_results[species]['errors'].append(f'Error loading {pt_file.name}: {str(e)}')

    validation_results[species]['valid_files'] = valid_count

    if valid_count == 0:
        all_valid = False

# 输出验证结果
print('Feature Validation Results:')
print('=' * 50)
for species, results in validation_results.items():
    status = '✓' if results['exists'] and results['valid_files'] > 0 else '✗'
    print(f'{status} {species.upper()}:')
    print(f'    Directory exists: {results[\"exists\"]}')
    print(f'    Feature files: {results[\"feature_files\"]}')
    print(f'    Valid files checked: {results[\"valid_files\"]}')
    if results['feature_dim']:
        print(f'    Feature dimension: {results[\"feature_dim\"]}')
    if results['errors']:
        print(f'    Errors: {results[\"errors\"]}')
    print()

exit(0 if all_valid else 1)
" | tee -a "$LOG_FILE"
}

# 主函数
main() {
    echo "$(date): Starting ESM-2 650M feature extraction pipeline..." | tee -a "$LOG_FILE"

    # 检查环境
    check_python_environment

    # 创建目录
    create_directories

    # 准备FASTA文件
    prepare_fasta_files

    # 提取特征
    if extract_all_features; then
        # 验证特征
        if validate_features; then
            echo "$(date): ESM-2 feature extraction pipeline completed successfully!" | tee -a "$LOG_FILE"
            exit 0
        else
            echo "$(date): Feature validation failed!" | tee -a "$LOG_FILE"
            exit 1
        fi
    else
        echo "$(date): Feature extraction failed!" | tee -a "$LOG_FILE"
        exit 1
    fi
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # 处理命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --species)
                SINGLE_SPECIES="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --validate-only)
                validate_features
                exit $?
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --species SPECIES    Extract features for single species (human/mouse/yeast)"
                echo "  --batch-size SIZE    Set batch size (default: $BATCH_SIZE)"
                echo "  --validate-only      Only validate existing features"
                echo "  --help               Show this help"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # 如果指定了单个物种，只处理该物种
    if [[ -n "$SINGLE_SPECIES" ]]; then
        if [[ -n "${SPECIES[$SINGLE_SPECIES]}" ]]; then
            echo "$(date): Processing single species: $SINGLE_SPECIES" | tee -a "$LOG_FILE"
            check_python_environment
            create_directories
            prepare_fasta_files
            if extract_esm_features "$SINGLE_SPECIES"; then
                echo "Feature extraction for $SINGLE_SPECIES completed successfully!"
                exit 0
            else
                echo "Feature extraction for $SINGLE_SPECIES failed!"
                exit 1
            fi
        else
            echo "Error: Unknown species '$SINGLE_SPECIES'. Available: ${!SPECIES[*]}"
            exit 1
        fi
    else
        # 运行完整流程
        main
    fi
fi
