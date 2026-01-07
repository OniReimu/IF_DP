#!/bin/bash
set -euo pipefail

# Local MPS-compatible sweep script for ablation_fast_no_calib.py
# Based on run_xiaorong.sh but adapted for single-machine MacBook Pro

# Fixed settings
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

# Sweep parameters
K_VALUES=(512 2048)
CLIP_RADII=(3.0 4.0)
USER_COUNTS=(10 100 200)
DP_EPOCHS_VALUES=(5 10 20)
DP_LR_VALUES=(1e-2 5e-2 1e-1)
REHEARSAL_LAMBDAS=(0.1 0.5 1.0)

# Output directory for logs
LOG_DIR="./results"
mkdir -p "${LOG_DIR}"

# Summary file for quick results
SUMMARY_FILE="${LOG_DIR}/sweep_summary.txt"
echo "Sweep started at: $(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')" > "${SUMMARY_FILE}"
echo "==================================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Create table header
printf "%-8s %-8s %-8s %-10s %-10s %-12s | %-18s %-18s %-18s %-18s | %-10s\n" \
  "k" "clip" "users" "dp_epochs" "dp_lr" "lambda" \
  "Vanilla_DP" "Vanilla_SAT" "Fisher" "Fisher_SAT" \
  "Status" >> "${SUMMARY_FILE}"
echo "-------------------------------------------------------------------------------------------------------------------------------------------------" >> "${SUMMARY_FILE}"

# Counter for tracking progress
total_runs=$((${#K_VALUES[@]} * ${#CLIP_RADII[@]} * ${#USER_COUNTS[@]} * ${#DP_EPOCHS_VALUES[@]} * ${#DP_LR_VALUES[@]} * ${#REHEARSAL_LAMBDAS[@]}))
current_run=0

echo "Starting sweep with ${total_runs} total configurations"
echo "Logs will be saved to ${LOG_DIR}"
echo "Summary will be written to ${SUMMARY_FILE}"
echo "=================================================="

# Nested loops for all parameter combinations
for k in "${K_VALUES[@]}"; do
  for clip in "${CLIP_RADII[@]}"; do
    for users in "${USER_COUNTS[@]}"; do
      for dp_epochs in "${DP_EPOCHS_VALUES[@]}"; do
        for dp_lr in "${DP_LR_VALUES[@]}"; do
          for rehearsal_lambda in "${REHEARSAL_LAMBDAS[@]}"; do
            current_run=$((current_run + 1))

            # Create descriptive tag for this run
            TAG="k${k}_clip${clip}_u${users}_e${dp_epochs}_lr${dp_lr}_lambda${rehearsal_lambda}"
            LOG_FILE="${LOG_DIR}/${TAG}.log"

            echo "[${current_run}/${total_runs}] Running: ${TAG}"
            echo "  Started at: $(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"

            # Run the experiment
            uv run ablation_fast_no_calib.py \
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

              # Extract all four ablation accuracies from the "Accuracy Comparison" section
              # Format: "   • Model_Name   :  acc% (excluded 0,1: x%, rest: y%)"
              vanilla_dp_acc=$(grep -A 10 "Accuracy Comparison" "${LOG_FILE}" | grep "Vanilla DP-SGD" | grep -v "DP-SAT" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
              vanilla_dpsat_acc=$(grep -A 10 "Accuracy Comparison" "${LOG_FILE}" | grep "Vanilla DP-SGD + DP-SAT" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
              fisher_normal_acc=$(grep -A 10 "Accuracy Comparison" "${LOG_FILE}" | grep "Fisher DP + Normal" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")
              fisher_dpsat_acc=$(grep -A 10 "Accuracy Comparison" "${LOG_FILE}" | grep "Fisher DP + DP-SAT" | grep -oE '[0-9]+\.[0-9]+%' | head -1 | sed 's/%//' || echo "N/A")

              # Extract MIA AUC* for each variant from the privacy tradeoff table
              # Log format varies by model name length:
              # "Vanilla DP-SGD": fields are [1][2] [3][4] [5=acc] [6=auc*] [7=adv]
              # "Vanilla DP-SGD + DP-SAT": fields are [1][2] [3][4][5][6] [7=acc] [8=auc*] [9=adv]
              # "Fisher DP + Normal": fields are [1][2] [3][4][5][6] [7=acc] [8=auc*] [9=adv]
              # "Fisher DP + DP-SAT": fields are [1][2] [3][4][5][6] [7=acc] [8=auc*] [9=adv]
              vanilla_dp_auc=$(grep -A 10 "Privacy vs Accuracy Tradeoff" "${LOG_FILE}" | grep "Vanilla DP-SGD" | grep -v "DP-SAT" | awk '{print $6}' | head -1 || echo "N/A")
              vanilla_dpsat_auc=$(grep -A 10 "Privacy vs Accuracy Tradeoff" "${LOG_FILE}" | grep "Vanilla DP-SGD + DP-SAT" | awk '{print $8}' | head -1 || echo "N/A")
              fisher_normal_auc=$(grep -A 10 "Privacy vs Accuracy Tradeoff" "${LOG_FILE}" | grep "Fisher DP + Normal" | awk '{print $8}' | head -1 || echo "N/A")
              fisher_dpsat_auc=$(grep -A 10 "Privacy vs Accuracy Tradeoff" "${LOG_FILE}" | grep "Fisher DP + DP-SAT" | awk '{print $8}' | head -1 || echo "N/A")

              # Format combined accuracy(auc) strings
              vanilla_dp_result="${vanilla_dp_acc} (${vanilla_dp_auc})"
              vanilla_dpsat_result="${vanilla_dpsat_acc} (${vanilla_dpsat_auc})"
              fisher_normal_result="${fisher_normal_acc} (${fisher_normal_auc})"
              fisher_dpsat_result="${fisher_dpsat_acc} (${fisher_dpsat_auc})"

              # Write table row
              printf "%-8s %-8s %-8s %-10s %-10s %-12s | %-18s %-18s %-18s %-18s | %-10s\n" \
                "${k}" "${clip}" "${users}" "${dp_epochs}" "${dp_lr}" "${rehearsal_lambda}" \
                "${vanilla_dp_result}" "${vanilla_dpsat_result}" "${fisher_normal_result}" "${fisher_dpsat_result}" \
                "SUCCESS" >> "${SUMMARY_FILE}"

              # Also print to console
              echo "  Results:"
              echo "    Vanilla_DP=${vanilla_dp_result} | Vanilla_SAT=${vanilla_dpsat_result}"
              echo "    Fisher=${fisher_normal_result} | Fisher_SAT=${fisher_dpsat_result}"
            else
              echo "  ✗ Failed with exit code ${exit_code}"
              echo "  Check log: ${LOG_FILE}"

              # Write failure to summary
              printf "%-8s %-8s %-8s %-10s %-10s %-12s | %-18s %-18s %-18s %-18s | %-10s\n" \
                "${k}" "${clip}" "${users}" "${dp_epochs}" "${dp_lr}" "${rehearsal_lambda}" \
                "FAILED" "FAILED" "FAILED" "FAILED" \
                "ERROR_${exit_code}" >> "${SUMMARY_FILE}"
            fi

            echo "  Finished at: $(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"
            echo "=================================================="

          done
        done
      done
    done
  done
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
