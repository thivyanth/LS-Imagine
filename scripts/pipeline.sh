#!/bin/bash
#
# LS-Imagine Full Pipeline: collection → affordance → finetune → train → eval
#
# Usage: ./scripts/pipeline.sh <task_name> [prompt] [--skip-affordance] [--skip-collect]
#
# Examples:
#   ./scripts/pipeline.sh harvest_log_in_plains "find and harvest a tree log"
# bash ./scripts/pipeline.sh mine_iron_ore "find and mine iron ore"
#   ./scripts/pipeline.sh mine_iron_ore --skip-affordance
#   ./scripts/pipeline.sh shear_sheep "shear sheep" --skip-collect --skip-affordance
#

set -e

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <task_name> [prompt] [--skip-affordance] [--skip-collect]"
  echo ""
  echo "Examples:"
  echo "  $0 harvest_log_in_plains 'find and harvest a tree log'"
  echo "  $0 mine_iron_ore --skip-affordance"
  echo "  $0 shear_sheep 'shear sheep' --skip-collect --skip-affordance"
  exit 1
fi

TASK=$1
shift

PROMPT=""
SKIP_AFFORDANCE=true
SKIP_COLLECT=true
EVAL_EPISODES=100

# Parse remaining args
while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-affordance)
      SKIP_AFFORDANCE=true
      shift
      ;;
    --skip-collect)
      SKIP_COLLECT=true
      shift
      ;;
    *)
      PROMPT="$1"
      shift
      ;;
  esac
done

echo "=========================================="
echo "LS-Imagine Pipeline"
echo "Task: $TASK"
echo "Prompt: ${PROMPT:-N/A}"
echo "Skip collect: $SKIP_COLLECT"
echo "Skip affordance: $SKIP_AFFORDANCE"
echo "=========================================="

# Stage 1: Collect rollouts
if [ "$SKIP_COLLECT" = false ]; then
  echo "[Stage 1/5] Collecting rollouts..."
  sh ./scripts/collect.sh "$TASK"
else
  echo "[Stage 1/5] SKIPPED: Collecting rollouts"
fi

# Stage 2: Generate affordance dataset
if [ "$SKIP_AFFORDANCE" = false ]; then
  if [ -z "$PROMPT" ]; then
    echo "Error: --prompt required when not skipping affordance generation"
    exit 1
  fi
  echo "[Stage 2/5] Generating affordance dataset..."
  sh ./scripts/affordance.sh "$TASK" "$PROMPT"
else
  echo "[Stage 2/5] SKIPPED: Generating affordance dataset"
fi

# Stage 3: Fine-tune U-Net
if [ "$SKIP_AFFORDANCE" = false ]; then
  echo "[Stage 3/5] Fine-tuning U-Net..."
  sh ./scripts/finetune_unet.sh "$TASK"
else
  echo "[Stage 3/5] SKIPPED: Fine-tuning U-Net (using existing weights)"
fi

# Stage 4: Train agent (world model + behavior)
echo "[Stage 4/5] Training world model and behavior..."
sh ./scripts/train.sh "$TASK"

# Stage 5: Evaluate
echo "[Stage 5/5] Evaluating agent..."
sh ./scripts/test.sh ./logdir/latest.pt "$EVAL_EPISODES" "$TASK"

echo "=========================================="
echo "Pipeline completed!"
echo "=========================================="


