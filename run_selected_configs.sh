#!/bin/bash
set -uo pipefail  # 移除了 -e，确保单个配置失败不中断整个扫描

# 已选配置扫描脚本 (ablation.py - 包含校准的完整版本)

# 固定设置
EPOCHS=100
TARGET_EPSILON=0.5
DELTA=1e-5
DP_LAYER="backbone.classifier"
MODEL_TYPE="efficientnet"
CALIBRATION_K=400
DP_SAT_MODE="fisher"
NON_IID="--non-iid"
EXCLUDE_CLASSES="0,1"
REG=10
COMBINED_STEPS=10
RUN_MIA="--run-mia"

# 检查 python 指令
PYTHON_CMD=$(command -v python3 || command -v python)

# 输出目录
LOG_DIR="./results"
mkdir -p "${LOG_DIR}"

# 汇总文件
SUMMARY_FILE="${LOG_DIR}/selected_configs_summary.txt"
echo "Selected configurations sweep started at: $(date)" > "${SUMMARY_FILE}"
echo "==================================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# 创建表头
printf "%-8s %-8s %-8s %-10s %-10s %-12s | %-18s %-18s %-18s %-18s %-18s %-18s | %-10s\n" \
  "k" "clip" "users" "dp_epochs" "dp_lr" "lambda" \
  "Vanilla_DP" "Vanilla_SAT" "Fisher" "Fisher_SAT" "Fisher+Calib" "FisherSAT+Calib" \
  "Status" >> "${SUMMARY_FILE}"
echo "---------------------------------------------------------------------------------------------------------------------------------------------------------------------" >> "${SUMMARY_FILE}"

# 选定的配置数组
# 格式: "k clip users dp_epochs dp_lr lambda"
CONFIGS=(
  "512 1.0 100 10 5e-2 0.1"
  "512 1.0 100 10 1e-1 1.0"
  "512 1.0 200 5 1e-1 0.1"
  "512 2.0 10 10 5e-2 1.0"
  "512 2.0 100 10 5e-2 0.1"
  "512 2.0 100 10 1e-1 0.1"
  "512 3.0 100 10 5e-2 0.5"
  "512 4.0 100 10 1e-2 0.1"
  "512 4.0 100 10 1e-2 0.5"
  "512 4.0 100 10 1e-2 1.0"
  "512 4.0 100 10 5e-2 0.5"
  "512 4.0 200 20 1e-2 0.5"
)

total_runs=${#CONFIGS[@]}
current_run=0

echo "Starting sweep with ${total_runs} selected configurations using $PYTHON_CMD"
echo "=================================================="

for config in "${CONFIGS[@]}"; do
  current_run=$((current_run + 1))
  read -r k clip users dp_epochs dp_lr rehearsal_lambda <<< "${config}"

  TAG="k${k}_clip${clip}_u${users}_e${dp_epochs}_lr${dp_lr}_lambda${rehearsal_lambda}"
  LOG_FILE="${LOG_DIR}/${TAG}_full.log"

  echo "[${current_run}/${total_runs}] Running: ${TAG}"

  # 运行实验 (移除了 uv run 和 --mps)
  $PYTHON_CMD ablation.py \
    --model-type "${MODEL_TYPE}" \
    --epochs "${EPOCHS}" \
    --target-epsilon "${TARGET_EPSILON}" \
    --delta "${DELTA}" \
    --dp-layer "${DP_LAYER}" \
    --calibration-k "${CALIBRATION_K}" \
    --dp-sat-mode "${DP_SAT_MODE}" \
    ${NON_IID} \
    --public-pretrain-exclude-classes "${EXCLUDE_CLASSES}" \
    --reg "${REG}" \
    --combined-steps "${COMBINED_STEPS}" \
    ${RUN_MIA} \
    --k "${k}" \
    --clip-radius "${clip}" \
    --users "${users}" \
    --dp-epochs "${dp_epochs}" \
    --shadow-epochs "${dp_epochs}" \
    --dp-lr "${dp_lr}" \
    --method public-fisher \
    --rehearsal-lambda "${rehearsal_lambda}" \
    2>&1 | tee "${LOG_FILE}"

  exit_code=${PIPESTATUS[0]}

  if [ ${exit_code} -eq 0 ]; then
    echo "  ✓ Completed successfully"

    # 指标提取逻辑 (grep 增强)
    vanilla_dp_acc=$(grep "Vanilla DP-SGD" "${LOG_FILE}" | grep -v "DP-SAT" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
    vanilla_dpsat_acc=$(grep "Vanilla DP-SGD + DP-SAT" "${LOG_FILE}" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
    fisher_normal_acc=$(grep "Fisher DP + Normal" "${LOG_FILE}" | grep -vE "DP-SAT|Calib" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
    fisher_dpsat_acc=$(grep "Fisher DP + DP-SAT" "${LOG_FILE}" | grep -v "Calib" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
    fisher_calib_acc=$(grep "Fisher DP + Normal + OPT Calib" "${LOG_FILE}" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
    fisher_dpsat_calib_acc=$(grep "Fisher DP + DP-SAT + OPT Calib" "${LOG_FILE}" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")

    vanilla_dp_auc=$(grep "Vanilla DP-SGD" "${LOG_FILE}" | grep -v "DP-SAT" | awk '{print $6}' | head -1 || echo "N/A")
    vanilla_dpsat_auc=$(grep "Vanilla DP-SGD + DP-SAT" "${LOG_FILE}" | awk '{print $8}' | head -1 || echo "N/A")
    fisher_normal_auc=$(grep "Fisher DP + Normal" "${LOG_FILE}" | grep -vE "DP-SAT|Calib" | awk '{print $8}' | head -1 || echo "N/A")
    fisher_dpsat_auc=$(grep "Fisher DP + DP-SAT" "${LOG_FILE}" | grep -v "Calib" | awk '{print $8}' | head -1 || echo "N/A")
    fisher_calib_auc=$(grep "Fisher DP + Normal + Calib" "${LOG_FILE}" | awk '{print $10}' | head -1 || echo "N/A")
    fisher_dpsat_calib_auc=$(grep "Fisher DP + DP-SAT + Calib" "${LOG_FILE}" | awk '{print $10}' | head -1 || echo "N/A")

    # 写入结果
    printf "%-8s %-8s %-8s %-10s %-10s %-12s | %-18s %-18s %-18s %-18s %-18s %-18s | %-10s\n" \
      "${k}" "${clip}" "${users}" "${dp_epochs}" "${dp_lr}" "${rehearsal_lambda}" \
      "${vanilla_dp_acc}(${vanilla_dp_auc})" "${vanilla_dpsat_acc}(${vanilla_dpsat_auc})" \
      "${fisher_normal_acc}(${fisher_normal_auc})" "${fisher_dpsat_acc}(${fisher_dpsat_auc})" \
      "${fisher_calib_acc}(${fisher_calib_auc})" "${fisher_dpsat_calib_acc}(${fisher_dpsat_calib_auc})" \
      "SUCCESS" >> "${SUMMARY_FILE}"
  else
    printf "%-8s %-8s %-8s %-10s %-10s %-12s | %-18s %-18s %-18s %-18s %-18s %-18s | %-10s\n" \
      "${k}" "${clip}" "${users}" "${dp_epochs}" "${dp_lr}" "${rehearsal_lambda}" \
      "ERR" "ERR" "ERR" "ERR" "ERR" "ERR" "FAIL_${exit_code}" >> "${SUMMARY_FILE}"
  fi
done

echo "Sweep complete! Summary at: ${SUMMARY_FILE}"