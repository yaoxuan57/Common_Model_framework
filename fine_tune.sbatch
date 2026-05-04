#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p h200p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=36G
#SBATCH --time=10:00:00
#SBATCH -J pl_ft
#SBATCH -o ~/logs/%x-%j.out

set -euo pipefail

# --- Initialize Environment Modules ---
if [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
  . /usr/share/Modules/init/bash
fi

# --- Load apptainer/singularity if available ---
if command -v module >/dev/null 2>&1; then
  module avail 2>&1 | grep -qi apptainer   && module load apptainer   || true
  module avail 2>&1 | grep -qi singularity && module load singularity || true
fi
APPTAINER_BIN="$(command -v apptainer || true)"
[ -z "$APPTAINER_BIN" ] && APPTAINER_BIN="$(command -v singularity || true)"
[ -z "$APPTAINER_BIN" ] && { echo "FATAL: apptainer/singularity not found"; exit 127; }

# ---- Paths ----
IMAGE="/scratch/prj0000000262/images/pytorch-2.4-cuda12.1.sif"
[ -f "$IMAGE" ] || { echo "FATAL: image not found: $IMAGE"; exit 2; }

export APPTAINER_CACHEDIR="/scratch/prj0000000262/.apptainer_cache"
mkdir -p "$APPTAINER_CACHEDIR" "$HOME/logs"

# Repo root (must contain requirements.txt and your python entry script)
CODE_DIR="$HOME/Taisin_FD/Taisin_FD"
DATA_DIR="/scratch/prj0000000262"      # adjust as needed
RUNS_DIR="/scratch/prj0000000262"    # where checkpoints/logs will go
RUN_NAME="Motor_1_trial_01"   # Alok to change
mkdir -p "$DATA_DIR" "$RUNS_DIR"

# Your python entrypoint file (the script that contains the code you pasted)
SCRIPT="fine_tune_taisin_motor_3.py"   # <- rename to your actual filename in repo

# ---- Args to your script ----
# Note: this script decides task_type from data_id (FEMTO => RUL else FD)
EXTRA_ARGS=(
  --data_path "$DATA_DIR"
  --data_id "Taisin_PHD_Historical_data"  # ALok to change
  --data_percentage "1"
  --model_id "TS_M5"

  --model_type "tiny"
  --patch_size 64
  --dropout 0.3
  --gpu_id 0
  --use_moe false

  --load_from_pretrained true
  --pretrained_model_dir "/home/users/astar/simt/ngyx/Taisin_FD/Taisin_FD/pretrained_models/Tiny"  # alok to change
  --pretraining_epoch_id 1
  --pretrained_model_type "normal"

  --num_epochs 200
  --patience 50
  --batch_size 8
  --lr 3e-4
  --wt_decay 1e-4
  --random_seed 42
  --task_type 'FD'
)

# ---- Logging ----
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] Job ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] GPUs: ${SLURM_GPUS:-N/A} | STEP_GPUS: ${SLURM_STEP_GPUS:-N/A} | GRES: ${SLURM_GRES:-N/A}"
echo "[INFO] IMAGE: $IMAGE"
echo "[INFO] CODE_DIR: $CODE_DIR"
echo "[INFO] DATA_DIR: $DATA_DIR"
echo "[INFO] RUNS_DIR: $RUNS_DIR"
echo "[ENV ] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# ---- Run manifest ----
cd "$CODE_DIR"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
mkdir -p "$RUN_DIR"
git rev-parse --short HEAD > "$RUN_DIR/git_commit.txt" 2>/dev/null || echo "no-git" > "$RUN_DIR/git_commit.txt"
printf "SLURM_JOB_ID=%s\nHOSTNAME=%s\n" "${SLURM_JOB_ID:-N/A}" "$(hostname)" > "$RUN_DIR/run_env.txt"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- Bind mounts ----
BIND_OPTS="--bind $CODE_DIR:/workspace --bind /scratch:/scratch --bind $HOME/logs:$HOME/logs"

# ---- Reusable venv & pip cache on scratch ----
VENV_DIR="/scratch/prj0000000262/ngyx/taisin/taisin-pl-py311"   # alok to change
PIP_CACHE_DIR="/scratch/prj0000000262/ngyx/taisin/.pip_cache"   # alok to change
mkdir -p "$(dirname "$VENV_DIR")" "$PIP_CACHE_DIR"

# ---- Serialize args safely ----
EXTRA_STR=$(printf "%q " "${EXTRA_ARGS[@]}")

# ---- Execute inside container ----
srun env \
  VENV_DIR="$VENV_DIR" \
  PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  SCRIPT="$SCRIPT" \
  RUN_DIR="$RUN_DIR" \
  RUN_NAME="$RUN_NAME" \
  EXTRA_STR="$EXTRA_STR" \
  "$APPTAINER_BIN" exec --nv $BIND_OPTS "$IMAGE" bash -lc '
set -euo pipefail
cd /workspace
echo "[IN ] CWD: $(pwd)"
ls -l requirements.txt || { echo "[ERR] requirements.txt not found at /workspace"; exit 3; }
ls -l "$SCRIPT" || { echo "[ERR] script not found at /workspace/$SCRIPT"; exit 4; }

python3 -V || python -V || true

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
. "$VENV_DIR/bin/activate"

export PIP_CACHE_DIR="$PIP_CACHE_DIR"
python -m pip install --upgrade pip wheel
pip install -r requirements.txt

# Optional: make logs/checkpoints land under RUN_DIR by running from there
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "[OK ] Starting training..."
python -u "/workspace/$SCRIPT" $EXTRA_STR
'

echo "[DONE] Outputs at: $RUN_DIR"
