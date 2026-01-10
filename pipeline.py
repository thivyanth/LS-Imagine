"""
This script is used to run the LS-Imagine pipeline.
python pipeline.py \
    --mode dreamer_baseline \
    --tasks harvest_sand
    
python pipeline.py \
    --mode ls_imagine \
    --tasks shear_sheep \
    --seed 1
"""

import subprocess
import os
import argparse
from datetime import datetime

# --- 1. CORE SETTINGS ---
os.environ["MINEDOJO_HEADLESS"] = "1"
TASKS = [
    # "harvest_log_in_plains", 
    "mine_iron_ore", 
    # "shear_sheep", 
    "harvest_water_with_bucket", 
    "harvest_sand"
         ]  # Add more: ["harvest_log_in_plains", "mine_iron_ore"]
STEPS = "1e6"

# --- 2. PIPELINE CONTROLS (Toggles) ---
SKIP_COLLECT = True
SKIP_AFFORDANCE = True
RUN_FINETUNE = False
RUN_TRAINING = True
RUN_EVAL = True
EVAL_EPISODES = 100

# --- 3. PROMPT DICTIONARY ---
PROMPTS = {
    # "harvest_log_in_plains": "find and harvest a tree log",
    "harvest_log_in_plains": "harvest log in plains",
    # "harvest_water_with_bucket": "obtain water",
    "harvest_water_with_bucket": "harvest water with a bucket",
    # "harvest_sand": "obtain sand",
    "harvest_sand": "harvest sand",
    "shear_sheep": "obtain wool",
    # "mine_iron_ore": "mine iron ore",
    "mine_iron_ore": "mine iron ore",
}

def run_cmd(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=os.environ.copy())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="dreamer_baseline",
        choices=["ls_imagine", "dreamer_baseline", "mpf_lsd", "mpf_lsd_baseline", "vlm_only_baseline"],
        help="Training mode (maps to a config section in configs.yaml).",
    )
    parser.add_argument("--tasks", nargs="+", default=TASKS, help="Tasks to run")
    parser.add_argument("--steps", type=str, default=STEPS, help="Training steps")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for training (forwarded to expr.py as --seed)")
    parser.add_argument("--envs", type=int, default=None, help="Number of envs (forwarded to expr.py/test.py as --envs)")
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Use process-based Parallel env wrapper (forwarded to expr.py/test.py as --parallel True)",
    )
    parser.add_argument(
        "--eval-envs",
        type=int,
        default=4,
        help="If set, overrides --envs for evaluation only (forwarded to test.py as --envs)",
    )
    parser.add_argument(
        "--eval-parallel",
        action="store_true",
        default=True,
        help="If set, enables parallel envs for evaluation only (forwarded to test.py as --parallel True)",
    )
    parser.add_argument(
        "--eval-checkpoint-dir",
        type=str,
        default=None,
        help="If set, evaluate this checkpoint directory (must contain latest.pt) instead of the run's logdir.",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default=None,
        choices=["ls_imagine", "dreamer_baseline"],
        help="If set, use this mode's config for evaluation (otherwise uses --mode).",
    )
    
    # Stage toggles
    parser.add_argument("--run-collect", action="store_true", default=not SKIP_COLLECT, help="Run collection stage")
    parser.add_argument("--run-affordance", action="store_true", default=not SKIP_AFFORDANCE, help="Run affordance stage")
    parser.add_argument("--run-finetune", action="store_true", default=RUN_FINETUNE, help="Run finetune stage")
    parser.add_argument("--skip-training", action="store_true", default=not RUN_TRAINING, help="Skip training stage")
    parser.add_argument("--skip-eval", action="store_true", default=not RUN_EVAL, help="Skip evaluation stage")
    
    args = parser.parse_args()
    
    MODE = args.mode
    steps_str = args.steps
    eval_episodes = args.eval_episodes

    # Baseline dreamer should physically disable affordance steps
    is_baseline = (MODE in ("dreamer_baseline", "mpf_lsd_baseline", "vlm_only_baseline"))
    
    for task in args.tasks:
        prompt = PROMPTS.get(task, "minecraft task")
        full_task = f"minedojo_{task}"
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        logdir = f"./logdir/{MODE}_{task}_{timestamp}"

        # Ensure W&B tags reflect the selected mode (used by plotting scripts).
        # W&B expects comma-separated tags in WANDB_TAGS.
        os.environ["WANDB_TAGS"] = "baseline" if is_baseline else "LS-Imagine"
        
        print("\n" + "="*60)
        print(f"STARTING PIPELINE FOR: {task}")
        print(f"MODE: {MODE} | PROMPT: {prompt}")
        print("="*60)

        # [Stage 1] Collect rollouts
        if args.run_collect:
            print("[1/5] Collecting rollouts...")
            run_cmd(["sh", "./scripts/collect.sh", task])
        else:
            print("[1/5] SKIPPED: Collect")

        # [Stage 2] Generate affordance dataset (Auto-skip if baseline)
        if args.run_affordance and not is_baseline:
            print("[2/5] Generating affordance dataset...")
            run_cmd(["sh", "./scripts/affordance.sh", task, prompt])
        else:
            reason = "baseline mode" if is_baseline else "not requested"
            print(f"[2/5] SKIPPED: Affordance (Reason: {reason})")

        # [Stage 3] Fine-tune U-Net (Auto-skip if baseline)
        if args.run_finetune and not is_baseline:
            print("[3/5] Fine-tuning U-Net...")
            run_cmd(["sh", "./scripts/finetune_unet.sh", task])
        else:
            reason = "baseline mode" if is_baseline else "not requested"
            print(f"[3/5] SKIPPED: Finetune (Reason: {reason})")

        # [Stage 4] Train agent
        if not args.skip_training:
            print("[4/5] Training agent...")
            cmd = [
                "xvfb-run", "-a", "python", "expr.py",
                "--mode", MODE,
                "--task", full_task,
                "--prompt", prompt,
                "--steps", str(steps_str),
                "--logdir", logdir
            ]
            if args.seed is not None:
                cmd += ["--seed", str(args.seed)]
            if args.envs is not None:
                cmd += ["--envs", str(args.envs)]
            if args.parallel:
                cmd += ["--parallel", "True"]
            run_cmd(cmd)
        else:
            print("[4/5] SKIPPED: Training")

        # [Stage 5] Evaluate
        if not args.skip_eval:
            print("[5/5] Evaluating agent...")
            checkpoint_dir = args.eval_checkpoint_dir or logdir
            checkpoint_file = os.path.join(checkpoint_dir, "latest.pt")
            if os.path.exists(checkpoint_file):
                eval_mode = args.eval_mode or MODE
                # If evaluation mode differs from training mode, keep tags consistent with eval.
                os.environ["WANDB_TAGS"] = "baseline" if (eval_mode == "dreamer_baseline") else "LS-Imagine"
                eval_envs = args.eval_envs if args.eval_envs is not None else args.envs
                eval_parallel = args.eval_parallel or args.parallel
                cmd = [
                    "xvfb-run", "-a", "python", "test.py",
                    "--mode", eval_mode,
                    "--task", full_task,
                    "--prompt", prompt,
                    "--logdir", "./logdir",
                    "--agent_checkpoint_dir", checkpoint_dir,
                    "--eval_episode_num", str(eval_episodes),
                ]
                if args.seed is not None:
                    cmd += ["--seed", str(args.seed)]
                if eval_envs is not None:
                    cmd += ["--envs", str(eval_envs)]
                if eval_parallel:
                    cmd += ["--parallel", "True"]
                run_cmd(cmd)
            else:
                print(f"Evaluation skipped: Checkpoint {checkpoint_file} not found.")
        else:
            print("[5/5] SKIPPED: Evaluation")

        print(f"\n>>> Completed Task: {task}")
        print("="*60)

if __name__ == "__main__":
    main()

