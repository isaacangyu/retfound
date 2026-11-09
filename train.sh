#!/usr/bin/env bash
# Train / benchmark RETFound on the OCT tasks.
#
#   bash train.sh                                   # one run with the defaults below
#   bash train.sh --adaptation finetune --eye OD    # override any default
#   bash train.sh --task pre-features,multi-binary --label soft,hard --gpus 0,1
#                                                   # 4 runs, spread over GPUs 0 and 1
#
# Any setting marked [sweep] takes a comma-separated list; every combination becomes one run.
# Spaces inside a value keep their meaning as a multi-value flag for that one run:
#   --eye "OS,OD OS"   ->  run 1: --eye OS      run 2: --eye OD OS
# Runs are queued and handed out to free GPU slots (--gpus x --jobs_per_gpu at a time).
# With more than one run, each run's output goes to logs/<timestamp>/<task>.log.
# Anything after "--" is passed straight to main_finetune.py (e.g. -- --blr 1e-3).
#
# Flags (default in brackets):
#   Model
#     --model        [sweep] mae | dinov2 | RETFound_mae | RETFound_dinov2 | Dinov2 | Dinov3   [mae]
#     --model_arch   [sweep] derived from --model when empty                                 []
#     --finetune     [sweep] weights; short names get the model prefix, e.g. natureOCT, meh,
#                            shanghai, natureCFP; empty = model's default (mae natureOCT, dinov2 meh)
#     --adaptation   [sweep] lp (frozen encoder, head only) | finetune (train all)           [lp]
#     --head         [sweep] mlp | linear                                                     [mlp]
#   Data / task
#     --data_path            sample store from prepare_data.py                               [./data]
#     --task         [sweep] pre-features | multi-binary | single-real | response            [pre-features]
#     --label        [sweep] hard | soft                                                     [hard]
#     --smoothing    [sweep] label smoothing eps for hard labels                             [0.1]
#     --eye          [sweep] OD | OS | "OD OS"                                               [OS]
#     --visit        [sweep] pre | post | "pre post"                                         [pre]
#     --image_idx    [sweep] e.g. 0, "0 1 2 3 4 5", or all (every slice; MIL)                [0]
#     --diagnosis    [sweep] 3 = DME (1 wAMD, 2 PCV, 4 RVO, 5 CME, 6 fellow, 9 other), 0 = all [3]
#     --anti_vegf    [sweep] e.g. 1; empty = all                                             []
#     --split_seed   [sweep] patient split seed (60/20/20)                                   [42]
#   Training
#     --epochs       [sweep]                                                                 [50]
#     --batch_size   [sweep]                                                                 [8]
#     --input_size                                                                           [224]
#     --num_workers          data loader workers per run                                     [0]
#     --hub_repo             HF repo for checkpoint + benchmark upload; "" = off             [angisaac/retfound-oct]
#   Batch
#     --gpus                 GPU ids "0,1,3", or a count "4" (= 0,1,2,3)                     [0]
#     --jobs_per_gpu         runs at the same time on each GPU                              [1]
#     --dry_run              print the runs and exit
#     -h | --help

# ==== Defaults ====
# Model settings
MODEL="mae"
MODEL_ARCH=""
FINETUNE=""
ADAPTATION="lp"
HEAD="mlp"
# Data settings
DATA_PATH="./data"
TASK="pre-features"
LABEL="hard"
SMOOTHING="0.1"
EYE="OS"
VISIT="pre"
IMAGE_IDX="0"
DIAGNOSIS="3"
ANTI_VEGF=""
SPLIT_SEED="42"
# Training settings
EPOCHS="50"
BATCH_SIZE="8"
INPUT_SIZE="224"
NUM_WORKERS="0"
HUB_REPO="angisaac/retfound-oct"
# Batch settings
GPUS="0"
JOBS_PER_GPU="1"
DRY_RUN=0

SWEEP_KEYS="MODEL MODEL_ARCH FINETUNE ADAPTATION HEAD TASK LABEL SMOOTHING EYE VISIT IMAGE_IDX DIAGNOSIS ANTI_VEGF SPLIT_SEED EPOCHS BATCH_SIZE"
OTHER_KEYS="DATA_PATH INPUT_SIZE NUM_WORKERS HUB_REPO GPUS JOBS_PER_GPU"

usage() { sed -n '2,/^# ==== Defaults/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

# ==== Parse flags ====
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry_run|--dry-run) DRY_RUN=1; shift; continue ;;
    --) shift; EXTRA=("$@"); break ;;
    --*=*) key="${1%%=*}"; val="${1#*=}"; shift ;;
    --*)   key="$1"; [ $# -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }; val="$2"; shift 2 ;;
    *) echo "Unknown argument: $1 (see bash train.sh --help)" >&2; exit 1 ;;
  esac
  key="${key#--}"; key="${key//-/_}"; key="${key^^}"
  case " ${SWEEP_KEYS} ${OTHER_KEYS} " in
    *" ${key} "*) printf -v "${key}" '%s' "${val}" ;;
    *) echo "Unknown flag: --${key,,} (see bash train.sh --help)" >&2; exit 1 ;;
  esac
done

# ==== Expand sweeps: one line per run, "KEY=value" fields separated by \x1f ====
SEP=$'\x1f'
RUNS=("")
for key in ${SWEEP_KEYS}; do
  IFS=',' read -ra vals <<< "${!key},"   # trailing comma keeps an empty value as one entry
  [ ${#vals[@]} -eq 0 ] && vals=("")
  next=()
  for r in "${RUNS[@]}"; do
    for v in "${vals[@]}"; do next+=("${r}${key}=${v}${SEP}"); done
  done
  RUNS=("${next[@]}")
done
# Which keys are swept (have more than one value) -> added to the run name
SWEPT=" "
for key in ${SWEEP_KEYS}; do [[ "${!key}" == *,* ]] && SWEPT="${SWEPT}${key} "; done

# Fill in one run's settings (sets globals in the calling subshell) and build its argument list.
load_run() {
  local field
  IFS="${SEP}" read -ra fields <<< "$1"
  for field in "${fields[@]}"; do printf -v "${field%%=*}" '%s' "${field#*=}"; done

  case "${MODEL}" in
    mae)    MODEL="RETFound_mae" ;;
    dinov2) MODEL="RETFound_dinov2" ;;
  esac
  if [ -z "${MODEL_ARCH}" ]; then
    case "${MODEL}" in
      RETFound_mae)    MODEL_ARCH="retfound_mae" ;;
      RETFound_dinov2) MODEL_ARCH="retfound_dinov2" ;;
      *) echo "--model_arch is required for --model ${MODEL}" >&2; return 1 ;;
    esac
  fi
  local default_ft=""
  case "${MODEL}" in
    RETFound_mae)    default_ft="RETFound_mae_natureOCT" ;;
    RETFound_dinov2) default_ft="RETFound_dinov2_meh" ;;
  esac
  [ -z "${FINETUNE}" ] && FINETUNE="${default_ft}"
  if [[ "${MODEL}" == RETFound_* && -n "${FINETUNE}" && "${FINETUNE}" != RETFound_* && "${FINETUNE}" != *.pth ]]; then
    FINETUNE="${MODEL}_${FINETUNE}"   # natureOCT -> RETFound_mae_natureOCT
  fi

  TASK="${TASK#--}"
  case "${LABEL#--}" in
    hard|hard-label) LABEL="hard-label" ;;
    soft|soft-label) LABEL="soft-label" ;;
    *) echo "--label must be hard or soft, got '${LABEL}'" >&2; return 1 ;;
  esac

  # Run name: same scheme as before, plus suffixes for settings that are swept or non-default
  RUN_NAME="${MODEL_ARCH}_${TASK}_${LABEL}_dx${DIAGNOSIS// /-}_${EYE// /}_${VISIT// /}_idx${IMAGE_IDX// /-}_${ADAPTATION}"
  [[ "${FINETUNE}" != "${default_ft}" || "${SWEPT}" == *" FINETUNE "* ]] && RUN_NAME="${RUN_NAME}_${FINETUNE#${MODEL}_}"
  [[ -n "${ANTI_VEGF}" ]]                                                && RUN_NAME="${RUN_NAME}_av${ANTI_VEGF// /-}"
  [[ "${HEAD}" != "mlp" || "${SWEPT}" == *" HEAD "* ]]                   && RUN_NAME="${RUN_NAME}_${HEAD}"
  [[ "${LABEL}" == "hard-label" && ( "${SMOOTHING}" != "0.1" || "${SWEPT}" == *" SMOOTHING "* ) ]] \
                                                                         && RUN_NAME="${RUN_NAME}_eps${SMOOTHING}"
  [[ "${SPLIT_SEED}" != "42" || "${SWEPT}" == *" SPLIT_SEED "* ]]        && RUN_NAME="${RUN_NAME}_seed${SPLIT_SEED}"
  [[ "${EPOCHS}" != "50" || "${SWEPT}" == *" EPOCHS "* ]]                && RUN_NAME="${RUN_NAME}_ep${EPOCHS}"
  [[ "${SWEPT}" == *" BATCH_SIZE "* ]]                                   && RUN_NAME="${RUN_NAME}_bs${BATCH_SIZE}"

  ARGS=(main_finetune.py
    --model "${MODEL}" --model_arch "${MODEL_ARCH}" --finetune "${FINETUNE}"
    --adaptation "${ADAPTATION}" --head "${HEAD}"
    "--${TASK}" "--${LABEL}" --smoothing "${SMOOTHING}"
    --eye ${EYE} --visit ${VISIT} --image_idx ${IMAGE_IDX} --split_seed "${SPLIT_SEED}"
    --savemodel --global_pool
    --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" --world_size 1 --epochs "${EPOCHS}"
    --data_path "${DATA_PATH}" --input_size "${INPUT_SIZE}" --task "${RUN_NAME}")
  [ -n "${DIAGNOSIS}" ] && ARGS+=(--diagnosis ${DIAGNOSIS})
  [ -n "${ANTI_VEGF}" ] && ARGS+=(--anti_vegf ${ANTI_VEGF})
  [ -n "${HUB_REPO}" ]  && ARGS+=(--hub_repo "${HUB_REPO}")
  ARGS+=("${EXTRA[@]}")
  return 0
}

# Check every run up front and catch duplicate names (they would overwrite each other).
declare -A SEEN
NAMES=()
for r in "${RUNS[@]}"; do
  name=$(load_run "$r" && echo "${RUN_NAME}") || exit 1
  [ -n "${SEEN[$name]}" ] && { echo "Two runs share the name ${name}; check your sweep values" >&2; exit 1; }
  SEEN[$name]=1; NAMES+=("${name}")
done

# ==== Launcher ====
# Windows PyTorch builds lack libuv, so torchrun fails there; one process per GPU doesn't need it,
# so call the retfound env's python directly. Works from Git Bash (/c/...) and WSL (/mnt/c/...).
# Override with PYTHON=/path/to/python.
RETFOUND_PY="/c/Users/isaac/anaconda3/envs/retfound/python.exe"
if   [ -n "${PYTHON}" ];             then LAUNCH="${PYTHON}"
elif [ -x "${RETFOUND_PY}" ];        then LAUNCH="${RETFOUND_PY}"
elif [ -x "/mnt${RETFOUND_PY}" ];    then LAUNCH="/mnt${RETFOUND_PY}"
elif command -v python >/dev/null 2>&1; then LAUNCH="python"
else LAUNCH="python3"
fi

# GPU slots: "--gpus 4" means 0,1,2,3; each GPU gets JOBS_PER_GPU slots
if [[ "${GPUS}" =~ ^[0-9]+$ && "${GPUS}" -gt 1 ]]; then GPUS=$(seq -s, 0 $((GPUS - 1))); fi
IFS=',' read -ra GPU_LIST <<< "${GPUS}"
SLOTS=()
for g in "${GPU_LIST[@]}"; do for _ in $(seq 1 "${JOBS_PER_GPU}"); do SLOTS+=("$g"); done; done

N=${#RUNS[@]}
echo "Launcher: ${LAUNCH}"
echo "${N} run(s) on GPU(s) ${GPUS} (${#SLOTS[@]} at a time):"
for i in "${!NAMES[@]}"; do echo "  [$((i + 1))/${N}] ${NAMES[$i]}"; done

if [ "${DRY_RUN}" = 1 ]; then
  for r in "${RUNS[@]}"; do (load_run "$r"; echo; echo "${LAUNCH} ${ARGS[*]}"); done
  exit 0
fi

# Single run: run in the foreground on the first GPU, output to the terminal.
if [ "${N}" -eq 1 ]; then
  load_run "${RUNS[0]}" || exit 1
  CUDA_VISIBLE_DEVICES="${SLOTS[0]}" ${LAUNCH} "${ARGS[@]}"
  exit $?
fi

# Several runs: a worker per slot pulls the next run from a shared counter (mkdir = atomic lock).
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
QUEUE="${LOG_DIR}/.queue"; echo 0 > "${QUEUE}"
echo "Logs: ${LOG_DIR}/"

next_run() {
  until mkdir "${QUEUE}.lock" 2>/dev/null; do sleep 0.2; done
  local i; i=$(cat "${QUEUE}"); echo $((i + 1)) > "${QUEUE}"
  rmdir "${QUEUE}.lock"
  echo "$i"
}

worker() {
  local gpu="$1" i rc
  while :; do
    i=$(next_run)
    [ "$i" -ge "${N}" ] && return 0
    (
      load_run "${RUNS[$i]}" || exit 1
      echo "[$(date +%H:%M:%S)] GPU ${gpu} start  [$((i + 1))/${N}] ${RUN_NAME}"
      CUDA_VISIBLE_DEVICES="${gpu}" ${LAUNCH} "${ARGS[@]}" > "${LOG_DIR}/${RUN_NAME}.log" 2>&1
      rc=$?
      if [ $rc -eq 0 ]; then echo "[$(date +%H:%M:%S)] GPU ${gpu} done   [$((i + 1))/${N}] ${RUN_NAME}"
      else echo "[$(date +%H:%M:%S)] GPU ${gpu} FAILED [$((i + 1))/${N}] ${RUN_NAME} (exit ${rc}, see ${LOG_DIR}/${RUN_NAME}.log)"
           echo "${RUN_NAME}" >> "${LOG_DIR}/failed.txt"; fi
    )
  done
}

trap 'echo "Stopping..."; kill $(jobs -p) 2>/dev/null; exit 130' INT TERM
for s in "${SLOTS[@]}"; do worker "$s" & done
wait

if [ -s "${LOG_DIR}/failed.txt" ]; then
  echo "$(wc -l < "${LOG_DIR}/failed.txt") of ${N} run(s) failed:"; sed 's/^/  /' "${LOG_DIR}/failed.txt"
  exit 1
fi
echo "All ${N} runs finished. Summary: benchmark/analysis.tsv"
