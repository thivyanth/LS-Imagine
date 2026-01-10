#!/bin/bash

if [ "$#" -lt 3 ]; then
  echo "Usage: ./scripts/test.sh <checkpoint_dir_containing_latest.pt> <eval_episode_num> <task_name_without_minedojo_prefix>"
  exit 1
fi

AGENT_CHECKPOINT_DIR=$1
EVAL_EPISODE_NUM=$2
TASK_NAME=$3
SEED=1

export MINEDOJO_HEADLESS=1
python test.py \
    --mode "${MODE:-ls_imagine}" \
    --task minedojo_${TASK_NAME} \
    --logdir ./logdir \
    --agent_checkpoint_dir ${AGENT_CHECKPOINT_DIR} \
    --eval_episode_num ${EVAL_EPISODE_NUM} \
    --seed ${SEED}
