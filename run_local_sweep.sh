#!/bin/bash
# 移除 set -e，防止单个运行失败导致整个 sweep 终止
set -uo pipefail

# 固定设置
EPOCHS=100
TARGET_EPSILON=0.5
DELTA=1e-5
DP_LAYER="backbone.head"
MODEL_TYPE="vit"
CALIBRATION_K=400
DP_SAT_MODE="fisher"
NON_IID="--non-iid"
EXCLUDE_CLASSES="0,1"
REG=10
COMBINED_STEPS=10
RUN_MIA="--run-mia"

# 参数搜索范围 (可以根据需要缩减，当前组合较多)
K_VALUES=(512 2048)
CLIP_RADII=(3.0 4.0)
USER_COUNTS=(10 100 200)
DP_EPOCHS_VALUES=(10 20)
DP_LR_VALUES=(1e-2 5e-2)
REHEARSAL_LAMBDAS=(0.5 1.0)

# 输出目录
LOG_DIR="./results"
mkdir -p "${LOG_DIR}"

SUMMARY_FILE="${LOG_DIR}/sweep_summary.txt"
echo "Sweep started at: $(date)" > "${SUMMARY_FILE}"
echo "==================================================" >> "${SUMMARY_FILE}"

# 表头
printf "%-8s %-8s %-8s %-10s %-10s %-12s | %-18s %-18s %-18s %-18s | %-10s\n" \
  "k" "clip" "users" "dp_epochs" "dp_lr" "lambda" \
  "Vanilla_DP" "Vanilla_SAT" "Fisher" "Fisher_SAT" \
  "Status" >> "${SUMMARY_FILE}"
echo "-------------------------------------------------------------------------------------------------------------------------------------------------" >> "${SUMMARY_FILE}"

total_runs=$((${#K_VALUES[@]} * ${#CLIP_RADII[@]} * ${#USER_COUNTS[@]} * ${#DP_EPOCHS_VALUES[@]} * ${#DP_LR_VALUES[@]} * ${#REHEARSAL_LAMBDAS[@]}))
current_run=0

# 检查 python 是否可用
PYTHON_CMD=$(command -v python3 || command -v python)

for k in "${K_VALUES[@]}"; do
  for clip in "${CLIP_RADII[@]}"; do
    for users in "${USER_COUNTS[@]}"; do
      for dp_epochs in "${DP_EPOCHS_VALUES[@]}"; do
        for dp_lr in "${DP_LR_VALUES[@]}"; do
          for rehearsal_lambda in "${REHEARSAL_LAMBDAS[@]}"; do
            current_run=$((current_run + 1))
            TAG="k${k}_clip${clip}_u${users}_e${dp_epochs}_lr${dp_lr}_lambda${rehearsal_lambda}"
            LOG_FILE="${LOG_DIR}/${TAG}.log"

            echo "[${current_run}/${total_runs}] Running: ${TAG}"

            # --- 关键修改：直接使用 python 运行 ---
            $PYTHON_CMD ablation_fast_no_calib.py \
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

            # 获取 python 进程的退出状态
            exit_code=${PIPESTATUS[0]}

            if [ ${exit_code} -eq 0 ]; then
              # 提取指标的正则逻辑保持不变...
              vanilla_dp_acc=$(grep "Vanilla DP-SGD" "${LOG_FILE}" | grep -v "DP-SAT" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
              vanilla_dpsat_acc=$(grep "Vanilla DP-SGD + DP-SAT" "${LOG_FILE}" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
              fisher_normal_acc=$(grep "Fisher DP + Normal" "${LOG_FILE}" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
              fisher_dpsat_acc=$(grep "Fisher DP + DP-SAT" "${LOG_FILE}" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")

              vanilla_dp_auc=$(grep "Vanilla DP-SGD" "${LOG_FILE}" | grep -v "DP-SAT" | awk '{print $6}' | head -1 || echo "N/A")
              vanilla_dpsat_auc=$(grep "Vanilla DP-SGD + DP-SAT" "${LOG_FILE}" | awk '{print $8}' | head -1 || echo "N/A")
              
              # 写入汇总
              printf "%-8s %-8s %-8s %-10s %-10s %-12s | %-18s %-18s %-18s %-18s | %-10s\n" \
                "${k}" "${clip}" "${users}" "${dp_epochs}" "${dp_lr}" "${rehearsal_lambda}" \
                "${vanilla_dp_acc}(${vanilla_dp_auc})" "${vanilla_dpsat_acc}(${vanilla_dpsat_auc})" "N/A" "N/A" \
                "SUCCESS" >> "${SUMMARY_FILE}"
            else
              printf "%-8s %-8s %-8s %-10s %-10s %-12s | %-18s %-18s %-18s %-18s | %-10s\n" \
                "${k}" "${clip}" "${users}" "${dp_epochs}" "${dp_lr}" "${rehearsal_lambda}" \
                "ERR" "ERR" "ERR" "ERR" \
                "FAIL_${exit_code}" >> "${SUMMARY_FILE}"
            fi
          done
        done
      done
    done
  done
done