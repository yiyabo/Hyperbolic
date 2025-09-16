#!/bin/bash

# STRING v12.0 数据下载脚本
# 下载Human, Mouse, Yeast的物理相互作用数据

set -e  # 遇到错误立即退出

# 配置
BASE_URL="https://stringdb-static.org/download"
VERSION="v12.0"
DATA_DIR="data/raw/string"
LOG_FILE="logs/download_string.log"

# 物种配置
declare -A SPECIES
SPECIES[human]=9606
SPECIES[mouse]=10090
SPECIES[yeast]=559292

# 文件类型配置
declare -a FILE_TYPES
FILE_TYPES=(
    "protein.physical.links"
    "protein.info"
    "protein.aliases"
    "protein.sequences"
)

# 置信度阈值
declare -a CONFIDENCE_LEVELS
CONFIDENCE_LEVELS=(700 900)

# 创建必要目录
mkdir -p "$DATA_DIR"
mkdir -p "logs"

# 记录开始时间
echo "$(date): Starting STRING data download..." | tee -a "$LOG_FILE"

# 下载函数
download_file() {
    local species_name=$1
    local taxid=$2
    local file_type=$3
    local confidence=${4:-""}

    local version_dir="${file_type}.${VERSION}"
    local extension="txt.gz"

    if [[ "$file_type" == "protein.sequences" ]]; then
        extension="fa.gz"
    fi

    if [[ -n "$confidence" && "$file_type" == "protein.physical.links" ]]; then
        local filename="${taxid}.${file_type}.${VERSION}.${extension}"
        local url="${BASE_URL}/${version_dir}/${filename}"
        # 注意: STRING下载页面支持置信度过滤，但需要手动设置参数
        echo "Note: Confidence filtering for ${confidence} should be done during processing" | tee -a "$LOG_FILE"
    else
        local filename="${taxid}.${file_type}.${VERSION}.${extension}"
        local url="${BASE_URL}/${version_dir}/${filename}"
    fi

    local output_path="${DATA_DIR}/${filename}"

    echo "$(date): Downloading ${species_name} ${file_type}..." | tee -a "$LOG_FILE"
    echo "  URL: ${url}" | tee -a "$LOG_FILE"
    echo "  Output: ${output_path}" | tee -a "$LOG_FILE"

    # 检查文件是否已存在
    if [[ -f "$output_path" ]]; then
        echo "  File already exists, checking size..." | tee -a "$LOG_FILE"
        local file_size=$(stat -c%s "$output_path" 2>/dev/null || stat -f%z "$output_path" 2>/dev/null || echo "0")
        if [[ "$file_size" -gt 1000 ]]; then
            echo "  File exists and appears complete, skipping download" | tee -a "$LOG_FILE"
            return 0
        else
            echo "  File exists but appears incomplete, re-downloading..." | tee -a "$LOG_FILE"
            rm -f "$output_path"
        fi
    fi

    # 下载文件
    local max_retries=3
    local retry_count=0

    while [[ $retry_count -lt $max_retries ]]; do
        if wget -c -t 3 -T 30 --progress=dot:giga -O "$output_path" "$url"; then
            echo "  Download completed successfully" | tee -a "$LOG_FILE"

            # 验证下载的文件
            if [[ -f "$output_path" ]]; then
                local downloaded_size=$(stat -c%s "$output_path" 2>/dev/null || stat -f%z "$output_path" 2>/dev/null || echo "0")
                echo "  Downloaded file size: ${downloaded_size} bytes" | tee -a "$LOG_FILE"

                # 简单验证：检查是否是gzip文件
                if file "$output_path" | grep -q "gzip"; then
                    echo "  File validation passed" | tee -a "$LOG_FILE"
                    return 0
                else
                    echo "  File validation failed: not a valid gzip file" | tee -a "$LOG_FILE"
                fi
            else
                echo "  Download failed: file not found" | tee -a "$LOG_FILE"
            fi
        else
            echo "  Download attempt $((retry_count + 1)) failed" | tee -a "$LOG_FILE"
        fi

        retry_count=$((retry_count + 1))
        if [[ $retry_count -lt $max_retries ]]; then
            local wait_time=$((retry_count * 5))
            echo "  Waiting ${wait_time} seconds before retry..." | tee -a "$LOG_FILE"
            sleep $wait_time
        fi
    done

    echo "  Error: Failed to download after $max_retries attempts" | tee -a "$LOG_FILE"
    return 1
}

# 检查wget是否可用
if ! command -v wget &> /dev/null; then
    echo "Error: wget is not installed. Please install wget first." | tee -a "$LOG_FILE"
    exit 1
fi

# 检查网络连接
if ! ping -c 1 string-db.org &> /dev/null; then
    echo "Warning: Cannot reach string-db.org, downloads may fail" | tee -a "$LOG_FILE"
fi

# 主下载循环
total_files=0
successful_downloads=0
failed_downloads=0

for species_name in "${!SPECIES[@]}"; do
    taxid=${SPECIES[$species_name]}
    echo "$(date): Processing species: ${species_name} (NCBI Taxonomy ID: ${taxid})" | tee -a "$LOG_FILE"

    for file_type in "${FILE_TYPES[@]}"; do
        total_files=$((total_files + 1))

        if download_file "$species_name" "$taxid" "$file_type"; then
            successful_downloads=$((successful_downloads + 1))
        else
            failed_downloads=$((failed_downloads + 1))
            echo "Warning: Failed to download ${species_name} ${file_type}" | tee -a "$LOG_FILE"
        fi
    done

    echo "$(date): Completed downloads for ${species_name}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

# 下载摘要
echo "$(date): Download Summary" | tee -a "$LOG_FILE"
echo "  Total files attempted: $total_files" | tee -a "$LOG_FILE"
echo "  Successful downloads: $successful_downloads" | tee -a "$LOG_FILE"
echo "  Failed downloads: $failed_downloads" | tee -a "$LOG_FILE"

# 检查下载的文件
echo "" | tee -a "$LOG_FILE"
echo "$(date): Downloaded files:" | tee -a "$LOG_FILE"
ls -lh "$DATA_DIR"/*.gz 2>/dev/null | while read line; do
    echo "  $line" | tee -a "$LOG_FILE"
done

# 计算总下载大小
if command -v du &> /dev/null; then
    total_size=$(du -sh "$DATA_DIR" | cut -f1)
    echo "  Total download size: $total_size" | tee -a "$LOG_FILE"
fi

# 创建下载完成标记文件
if [[ $failed_downloads -eq 0 ]]; then
    touch "${DATA_DIR}/.download_complete"
    echo "$(date): All downloads completed successfully!" | tee -a "$LOG_FILE"

    # 创建README文件说明下载的数据
    cat > "${DATA_DIR}/README.md" << EOF
# STRING v12.0 Downloaded Data

Downloaded on: $(date)
Script: 00_download_string.sh

## Species Data

### Human (Homo sapiens, NCBI Taxonomy ID: 9606)
- protein.physical.links.v12.0.txt.gz: Physical interaction links
- protein.info.v12.0.txt.gz: Protein information
- protein.aliases.v12.0.txt.gz: Protein aliases for ID mapping
- protein.sequences.v12.0.fa.gz: Protein sequences

### Mouse (Mus musculus, NCBI Taxonomy ID: 10090)
- protein.physical.links.v12.0.txt.gz: Physical interaction links
- protein.info.v12.0.txt.gz: Protein information
- protein.aliases.v12.0.txt.gz: Protein aliases for ID mapping
- protein.sequences.v12.0.fa.gz: Protein sequences

### Yeast (Saccharomyces cerevisiae, NCBI Taxonomy ID: 559292)
- protein.physical.links.v12.0.txt.gz: Physical interaction links
- protein.info.v12.0.txt.gz: Protein information
- protein.aliases.v12.0.txt.gz: Protein aliases for ID mapping
- protein.sequences.v12.0.fa.gz: Protein sequences

## Usage Notes

1. Physical links files contain confidence scores (combined_score)
2. Recommend filtering with confidence >= 700 or >= 900
3. Use protein.aliases files for ID mapping to UniProt
4. All files are gzip compressed

## Data Source

STRING Database v12.0
https://string-db.org/
https://stringdb-downloads.org/

EOF

    exit 0
else
    echo "$(date): Some downloads failed. Please check the log and retry." | tee -a "$LOG_FILE"
    exit 1
fi
