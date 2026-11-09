#!/usr/bin/env bash
# Run a hand-written list of train.sh runs, queued over the GPUs below.
#
#   bash script.sh              # run everything in RUNS
#   bash script.sh --dry_run    # print what would run
#
# Each line in RUNS is the flags for one `bash train.sh` call (see `bash train.sh --help`);
# unset flags use train.sh's defaults. Keep each line to a single run (no comma sweeps) --
# use train.sh directly for sweeps.

# ==== Resources ====
GPUS="0"            # GPU ids, e.g. "0,1,3"
JOBS_PER_GPU=1      # runs at the same time on each GPU (ViT-L lp at bs 8 needs ~3-4 GB each)
NUM_WORKERS=0       # CPU data-loader processes per run (CPU cores used per run ~= this + 1)

# ==== Runs (one per line) ====
RUNS=(
  "--task pre-features --label hard --eye \"OD OS\" --image_idx all"
  "--task pre-features --label soft --eye \"OD OS\" --image_idx all"
  "--task multi-binary --label soft --eye \"OD OS\" --image_idx all"
  "--task single-real --eye \"OD OS\" --image_idx all"
  "--task response --label hard --eye \"OD OS\" --image_idx all"
  "--task response --label soft --eye \"OD OS\" --image_idx all"
)

# Shared flags added to every run (e.g. "--hub_repo ''" to skip uploads, "--epochs 30")
COMMON=""

# ==== Queue (no need to edit below) ====
DRY_RUN=0; [ "$1" = "--dry_run" ] || [ "$1" = "--dry-run" ] && DRY_RUN=1
cd "$(dirname "$0")"

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
SLOTS=(); for g in "${GPU_LIST[@]}"; do for _ in $(seq 1 "${JOBS_PER_GPU}"); do SLOTS+=("$g"); done; done
N=${#RUNS[@]}
LOG_DIR="logs/script_$(date +%Y%m%d_%H%M%S)"

echo "${N} run(s) on GPU(s) ${GPUS}, ${JOBS_PER_GPU} per GPU (${#SLOTS[@]} at a time), ${NUM_WORKERS} loader workers each"
for i in "${!RUNS[@]}"; do echo "  [$((i + 1))/${N}] ${RUNS[$i]}"; done
if [ "${DRY_RUN}" = 1 ]; then
  for r in "${RUNS[@]}"; do eval "bash train.sh $r ${COMMON} --num_workers ${NUM_WORKERS} --dry_run" | tail -1; done
  exit 0
fi

mkdir -p "${LOG_DIR}"; QUEUE="${LOG_DIR}/.queue"; echo 0 > "${QUEUE}"
echo "Logs: ${LOG_DIR}/"

next_run() {
  until mkdir "${QUEUE}.lock" 2>/dev/null; do sleep 0.2; done
  local i; i=$(cat "${QUEUE}"); echo $((i + 1)) > "${QUEUE}"; rmdir "${QUEUE}.lock"; echo "$i"
}

worker() {
  local gpu="$1" i rc log
  while :; do
    i=$(next_run); [ "$i" -ge "${N}" ] && return 0
    log="${LOG_DIR}/run$((i + 1)).log"
    echo "[$(date +%H:%M:%S)] GPU ${gpu} start  [$((i + 1))/${N}] ${RUNS[$i]}"
    eval "bash train.sh ${RUNS[$i]} ${COMMON} --num_workers ${NUM_WORKERS} --gpus ${gpu}" > "${log}" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then echo "[$(date +%H:%M:%S)] GPU ${gpu} done   [$((i + 1))/${N}]"
    else echo "[$(date +%H:%M:%S)] GPU ${gpu} FAILED [$((i + 1))/${N}] (exit ${rc}, see ${log})"
         echo "[$((i + 1))] ${RUNS[$i]}" >> "${LOG_DIR}/failed.txt"; fi
  done
}

trap 'echo "Stopping..."; kill $(jobs -p) 2>/dev/null; exit 130' INT TERM
for s in "${SLOTS[@]}"; do worker "$s" & done
wait

if [ -s "${LOG_DIR}/failed.txt" ]; then
  echo "$(wc -l < "${LOG_DIR}/failed.txt") of ${N} run(s) failed:"; sed 's/^/  /' "${LOG_DIR}/failed.txt"; exit 1
fi
echo "All ${N} runs finished. Summary: benchmark/analysis.tsv"
