#!/bin/bash
set -euo pipefail

# Selected configurations sweep script for ablation.py (full ablation with calibration)
# Based on run_local_sweep.sh but for specific parameter combinations

# Fixed settings (same as run_local_sweep.sh)
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

# Output directory for logs
LOG_DIR="./results"
mkdir -p "${LOG_DIR}"

# Summary file for quick results
SUMMARY_FILE="${LOG_DIR}/selected_configs_summary.txt"
echo "Selected configurations sweep started at: $(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')" > "${SUMMARY_FILE}"
echo "==================================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Create table header
printf "%-8s %-8s %-8s %-10s %-10s %-12s | %-18s %-18s %-18s %-18s %-18s %-18s | %-10s\n" \
  "k" "clip" "users" "dp_epochs" "dp_lr" "lambda" \
  "Vanilla_DP" "Vanilla_SAT" "Fisher" "Fisher_SAT" "Fisher+Calib" "FisherSAT+Calib" \
  "Status" >> "${SUMMARY_FILE}"
echo "---------------------------------------------------------------------------------------------------------------------------------------------------------------------" >> "${SUMMARY_FILE}"

# Define selected configurations as an array
# Format: "k clip users dp_epochs dp_lr lambda"
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

echo "Starting sweep with ${total_runs} selected configurations"
echo "Logs will be saved to ${LOG_DIR}"
echo "Summary will be written to ${SUMMARY_FILE}"
echo "=================================================="

# Loop through selected configurations
for config in "${CONFIGS[@]}"; do
  current_run=$((current_run + 1))

  # Parse configuration
  read -r k clip users dp_epochs dp_lr rehearsal_lambda <<< "${config}"

  # Create descriptive tag for this run
  TAG="k${k}_clip${clip}_u${users}_e${dp_epochs}_lr${dp_lr}_lambda${rehearsal_lambda}"
  LOG_FILE="${LOG_DIR}/${TAG}_full.log"

  echo "[${current_run}/${total_runs}] Running: ${TAG}"
  echo "  Config: k=${k} clip=${clip} users=${users} dp_epochs=${dp_epochs} dp_lr=${dp_lr} lambda=${rehearsal_lambda}"
  echo "  Started at: $(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"

  # Run the experiment with ablation.py (full version with calibration)
  uv run ablation.py \
    --mps \
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
    --rehearsal-lambda "${rehearsal_lambda}" \
    2>&1 | tee "${LOG_FILE}"

  exit_code=${PIPESTATUS[0]}

  # Extract key metrics from the log file
  if [ ${exit_code} -eq 0 ]; then
    echo "  ✓ Completed successfully"

    # Extract all six ablation accuracies from the "Accuracy Comparison" section
    # Format: "   • Model_Name   :  acc% (excluded 0,1: x%, rest: y%)"
    vanilla_dp_acc=$(grep -A 15 "Accuracy Comparison" "${LOG_FILE}" | grep "Vanilla DP-SGD" | grep -v "DP-SAT" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
    vanilla_dpsat_acc=$(grep -A 15 "Accuracy Comparison" "${LOG_FILE}" | grep "Vanilla DP-SGD + DP-SAT" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
    fisher_normal_acc=$(grep -A 15 "Accuracy Comparison" "${LOG_FILE}" | grep "Fisher DP + Normal" | grep -v "OPT Calib" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
    fisher_dpsat_acc=$(grep -A 15 "Accuracy Comparison" "${LOG_FILE}" | grep "Fisher DP + DP-SAT" | grep -v "OPT Calib" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
    fisher_calib_acc=$(grep -A 15 "Accuracy Comparison" "${LOG_FILE}" | grep "Fisher DP + Normal + OPT Calib" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
    fisher_dpsat_calib_acc=$(grep -A 15 "Accuracy Comparison" "${LOG_FILE}" | grep "Fisher DP + DP-SAT + OPT Calib" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")

    # Extract MIA AUC* for each variant from the privacy tradeoff table
    # Log format varies by model name length:
    # "Vanilla DP-SGD": fields are [1][2] [3][4] [5=acc] [6=auc*] [7=adv]
    # "Vanilla DP-SGD + DP-SAT": fields are [1][2] [3][4][5][6] [7=acc] [8=auc*] [9=adv]
    # "Fisher DP + Normal": fields are [1][2] [3][4][5][6] [7=acc] [8=auc*] [9=adv]
    # "Fisher DP + DP-SAT": fields are [1][2] [3][4][5][6] [7=acc] [8=auc*] [9=adv]
    # "Fisher DP + Normal + Calib": fields are [1][2] [3][4][5][6][7][8] [9=acc] [10=auc*] [11=adv]
    # "Fisher DP + DP-SAT + Calib": fields are [1][2] [3][4][5][6][7][8] [9=acc] [10=auc*] [11=adv]
    vanilla_dp_auc=$(grep -A 15 "Privacy vs Accuracy Tradeoff" "${LOG_FILE}" | grep "Vanilla DP-SGD" | grep -v "DP-SAT" | awk '{print $6}' | head -1 || echo "N/A")
    vanilla_dpsat_auc=$(grep -A 15 "Privacy vs Accuracy Tradeoff" "${LOG_FILE}" | grep "Vanilla DP-SGD + DP-SAT" | awk '{print $8}' | head -1 || echo "N/A")
    fisher_normal_auc=$(grep -A 15 "Privacy vs Accuracy Tradeoff" "${LOG_FILE}" | grep "Fisher DP + Normal" | grep -v "Calib" | awk '{print $8}' | head -1 || echo "N/A")
    fisher_dpsat_auc=$(grep -A 15 "Privacy vs Accuracy Tradeoff" "${LOG_FILE}" | grep "Fisher DP + DP-SAT" | grep -v "Calib" | awk '{print $8}' | head -1 || echo "N/A")
    fisher_calib_auc=$(grep -A 15 "Privacy vs Accuracy Tradeoff" "${LOG_FILE}" | grep "Fisher DP + Normal + Calib" | awk '{print $10}' | head -1 || echo "N/A")
    fisher_dpsat_calib_auc=$(grep -A 15 "Privacy vs Accuracy Tradeoff" "${LOG_FILE}" | grep "Fisher DP + DP-SAT + Calib" | awk '{print $10}' | head -1 || echo "N/A")

    # Format combined accuracy(auc) strings
    vanilla_dp_result="${vanilla_dp_acc} (${vanilla_dp_auc})"
    vanilla_dpsat_result="${vanilla_dpsat_acc} (${vanilla_dpsat_auc})"
    fisher_normal_result="${fisher_normal_acc} (${fisher_normal_auc})"
    fisher_dpsat_result="${fisher_dpsat_acc} (${fisher_dpsat_auc})"
    fisher_calib_result="${fisher_calib_acc} (${fisher_calib_auc})"
    fisher_dpsat_calib_result="${fisher_dpsat_calib_acc} (${fisher_dpsat_calib_auc})"

    # Write table row
    printf "%-8s %-8s %-8s %-10s %-10s %-12s | %-18s %-18s %-18s %-18s %-18s %-18s | %-10s\n" \
      "${k}" "${clip}" "${users}" "${dp_epochs}" "${dp_lr}" "${rehearsal_lambda}" \
      "${vanilla_dp_result}" "${vanilla_dpsat_result}" "${fisher_normal_result}" "${fisher_dpsat_result}" \
      "${fisher_calib_result}" "${fisher_dpsat_calib_result}" \
      "SUCCESS" >> "${SUMMARY_FILE}"

    # Also print to console
    echo "  Results:"
    echo "    Vanilla_DP=${vanilla_dp_result} | Vanilla_SAT=${vanilla_dpsat_result}"
    echo "    Fisher=${fisher_normal_result} | Fisher_SAT=${fisher_dpsat_result}"
    echo "    Fisher+Calib=${fisher_calib_result} | FisherSAT+Calib=${fisher_dpsat_calib_result}"
  else
    echo "  ✗ Failed with exit code ${exit_code}"
    echo "  Check log: ${LOG_FILE}"

    # Write failure to summary
    printf "%-8s %-8s %-8s %-10s %-10s %-12s | %-18s %-18s %-18s %-18s %-18s %-18s | %-10s\n" \
      "${k}" "${clip}" "${users}" "${dp_epochs}" "${dp_lr}" "${rehearsal_lambda}" \
      "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" "FAILED" \
      "ERROR_${exit_code}" >> "${SUMMARY_FILE}"
  fi

  echo "  Finished at: $(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "=================================================="

done

echo "==================================================" >> "${SUMMARY_FILE}"
echo "Sweep completed at: $(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')" >> "${SUMMARY_FILE}"
echo "Total runs: ${current_run}/${total_runs}" >> "${SUMMARY_FILE}"

echo ""
echo "Sweep complete! All logs saved in ${LOG_DIR}"
echo "Total runs: ${current_run}/${total_runs}"
echo ""
echo "Quick summary written to: ${SUMMARY_FILE}"
echo "To monitor progress in real-time, run:"
echo "  tail -f ${SUMMARY_FILE}"
