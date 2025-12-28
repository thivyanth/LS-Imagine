import subprocess
import os
import argparse
from datetime import datetime

# --- 1. CORE SETTINGS ---
os.environ["MINEDOJO_HEADLESS"] = "1"
TASKS = [
    "harvest_log_in_plains", 
    # "mine_iron_ore", 
    # "shear_sheep", 
    # "harvest_water_with_bucket", 
    # "harvest_sand"
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
    "harvest_log_in_plains": "find and harvest a tree log",
    "harvest_water_with_bucket": "obtain water",
    "harvest_sand": "obtain sand",
    "shear_sheep": "obtain wool",
    "mine_iron_ore": "mine iron ore",
}

def run_cmd(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="dreamer_baseline", choices=["ls_imagine", "dreamer_baseline"], help="Training mode")
    parser.add_argument("--tasks", nargs="+", default=TASKS, help="Tasks to run")
    parser.add_argument("--steps", type=str, default=STEPS, help="Training steps")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES, help="Evaluation episodes")
    
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
    is_baseline = (MODE == "dreamer_baseline")
    
    for task in args.tasks:
        prompt = PROMPTS.get(task, "minecraft task")
        full_task = f"minedojo_{task}"
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        logdir = f"./logdir/{MODE}_{task}_{timestamp}"
        
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
            run_cmd([
                "xvfb-run", "-a", "python", "expr.py",
                "--configs", "minedojo", MODE,
                "--task", full_task,
                "--steps", str(steps_str),
                "--logdir", logdir
            ])
        else:
            print("[4/5] SKIPPED: Training")

        # [Stage 5] Evaluate
        if not args.skip_eval:
            print("[5/5] Evaluating agent...")
            checkpoint = os.path.join(logdir, "latest.pt")
            if os.path.exists(checkpoint):
                run_cmd([
                    "sh", "./scripts/test.sh",
                    checkpoint, str(eval_episodes), full_task
                ])
            else:
                print(f"Evaluation skipped: Checkpoint {checkpoint} not found.")
        else:
            print("[5/5] SKIPPED: Evaluation")

        print(f"\n>>> Completed Task: {task}")
        print("="*60)

if __name__ == "__main__":
    main()

