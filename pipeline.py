import subprocess
import os
from datetime import datetime

# --- 1. CORE SETTINGS ---
MODE = "dreamer_baseline"  # "ls_imagine" or "dreamer_baseline"
# MODE = "ls_imagine"
TASKS = [
    "harvest_log_in_plains", 
    "mine_iron_ore", 
    "shear_sheep", 
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
    # Baseline dreamer should physically disable affordance steps
    is_baseline = (MODE == "dreamer_baseline")
    
    for task in TASKS:
        prompt = PROMPTS.get(task, "minecraft task")
        full_task = f"minedojo_{task}"
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        logdir = f"./logdir/{MODE}_{task}_{timestamp}"
        
        print("\n" + "="*60)
        print(f"STARTING PIPELINE FOR: {task}")
        print(f"MODE: {MODE} | PROMPT: {prompt}")
        print("="*60)

        # [Stage 1] Collect rollouts
        if not SKIP_COLLECT:
            print("[1/5] Collecting rollouts...")
            run_cmd(["sh", "./scripts/collect.sh", task])
        else:
            print("[1/5] SKIPPED: Collect")

        # [Stage 2] Generate affordance dataset (Auto-skip if baseline)
        if not SKIP_AFFORDANCE and not is_baseline:
            print("[2/5] Generating affordance dataset...")
            run_cmd(["sh", "./scripts/affordance.sh", task, prompt])
        else:
            reason = "baseline mode" if is_baseline else "user toggle"
            print(f"[2/5] SKIPPED: Affordance (Reason: {reason})")

        # [Stage 3] Fine-tune U-Net (Auto-skip if baseline)
        if RUN_FINETUNE and not is_baseline:
            print("[3/5] Fine-tuning U-Net...")
            run_cmd(["sh", "./scripts/finetune_unet.sh", task])
        else:
            reason = "baseline mode" if is_baseline else "user toggle"
            print(f"[3/5] SKIPPED: Finetune (Reason: {reason})")

        # [Stage 4] Train agent
        if RUN_TRAINING:
            print("[4/5] Training agent...")
            run_cmd([
                "python", "expr.py",
                "--configs", "minedojo", MODE,
                "--task", full_task,
                "--steps", str(STEPS),
                "--logdir", logdir
            ])
        else:
            print("[4/5] SKIPPED: Training")

        # [Stage 5] Evaluate
        if RUN_EVAL:
            print("[5/5] Evaluating agent...")
            checkpoint = os.path.join(logdir, "latest.pt")
            if os.path.exists(checkpoint):
                run_cmd([
                    "sh", "./scripts/test.sh",
                    checkpoint, str(EVAL_EPISODES), full_task
                ])
            else:
                print(f"Evaluation skipped: Checkpoint {checkpoint} not found.")
        else:
            print("[5/5] SKIPPED: Evaluation")

        print(f"\n>>> Completed Task: {task}")
        print("="*60)

if __name__ == "__main__":
    main()

