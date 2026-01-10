#!/bin/bash
#
# Flexible DreamerV3 / LS-Imagine Pipeline
# Controls all 5 stages: Collect -> Affordance -> Finetune -> Train -> Eval

set -e

# --- 1. CORE SETTINGS ---
MODE="dreamer_baseline" # "ls_imagine" or "dreamer_baseline"
TASKS=("harvest_log_in_plains") # Example: ("harvest_log_in_plains" "mine_iron_ore")
STEPS="1e6"
EVAL_EPISODES=100

# --- 2. STEP TOGGLES (Set to true/false) ---
SKIP_COLLECT=true
SKIP_AFFORDANCE=true
RUN_FINETUNE=false
RUN_TRAINING=true
RUN_EVAL=true

# --- 3. PROMPT DICTIONARY ---
get_prompt() {
  case $1 in
    "harvest_log_in_plains")     echo "find and harvest a tree log" ;;
    "harvest_water_with_bucket") echo "obtain water" ;;
    "harvest_sand")              echo "obtain sand" ;;
    "shear_sheep")               echo "obtain wool" ;;
    "mine_iron_ore")             echo "mine iron ore" ;;
    *)                           echo "minecraft task" ;;
  esac
}

# --- PIPELINE EXECUTION ---
for TASK in "${TASKS[@]}"; do
  PROMPT=$(get_prompt "$TASK")
  FULL_TASK="minedojo_${TASK}"
  TIMESTAMP=$(date +%Y%m%dT%H%M%S)
  LOGDIR="./logdir/${MODE}_${TASK}_${TIMESTAMP}"

  echo "=========================================="
  echo "STARTING PIPELINE FOR: $TASK"
  echo "MODE: $MODE | PROMPT: $PROMPT"
  echo "=========================================="

  # [Stage 1] Collect rollouts
  if [ "$SKIP_COLLECT" = false ]; then
    echo "[1/5] Collecting rollouts..."
    sh ./scripts/collect.sh "$TASK"
  fi

  # [Stage 2] Generate affordance dataset
  if [ "$SKIP_AFFORDANCE" = false ]; then
    echo "[2/5] Generating affordance dataset..."
    sh ./scripts/affordance.sh "$TASK" "$PROMPT"
  fi

  # [Stage 3] Fine-tune U-Net
  if [ "$RUN_FINETUNE" = true ]; then
    echo "[3/5] Fine-tuning U-Net..."
    sh ./scripts/finetune_unet.sh "$TASK"
  fi

  # [Stage 4] Train agent
  if [ "$RUN_TRAINING" = true ]; then
    echo "[4/5] Training agent..."
    python expr.py \
        --mode "$MODE" \
        --task "$FULL_TASK" \
        --steps "$STEPS" \
        --logdir "$LOGDIR"
  fi

  # [Stage 5] Evaluate
  if [ "$RUN_EVAL" = true ]; then
    echo "[5/5] Evaluating agent..."
    # Look for latest.pt in the current LOGDIR
    CHECKPOINT="${LOGDIR}/latest.pt"
    if [ -f "$CHECKPOINT" ]; then
        sh ./scripts/test.sh "$CHECKPOINT" "$EVAL_EPISODES" "$FULL_TASK"
    else
        echo "Evaluation skipped: Checkpoint $CHECKPOINT not found."
    fi
  fi

  echo ">>> Completed Task: $TASK"
  echo "=========================================="
done
