# MineDojo tasks and world generation
_Exported on 1/5/2026 at 20:44:11 GMT+5:30 from Cursor (2.2.44)_

---

**User**

explain the task deatils of the minedojo 5 tasks
how the minecraft world is generated 

---

**Cursor**

Got it. I’m going to (1) search this repo for where the **five MineDojo tasks** are defined/configured, (2) summarize each task’s goal, reward/success conditions, and key environment settings, and then (3) trace how the **Minecraft world generation** (seed/biome/structures/reset) is set up in code so the explanation matches *your exact setup* (not generic MineDojo docs).

Nice — this repo already “pins down” the **exact 5 tasks** you mean (the LS-Imagine paper set): `harvest_log_in_plains`, `harvest_water_with_bucket`, `harvest_sand`, `shear_sheep`, `mine_iron_ore`. Next I’ll pull their YAML specs (success/terminal/reward/world settings) and then trace the MineDojo reset/world-gen path so the explanation matches your configuration.

Good catch: the wrappers live under `LS-Imagine/envs/tasks/base/` (not a single `base.py`). Next I’ll open the core wrappers (`reward_wrapper.py`, `success_wrapper.py`, `terminal_wrapper.py`) so we can precisely describe reward shaping + success/termination for the 5 tasks.

### The 5 MineDojo tasks (as configured in *this repo*)
These are defined in `LS-Imagine/envs/tasks/task_specs.yaml` and used as `minedojo_<task>` (see `LS-Imagine/pipeline.py`).

For all 5 tasks, the **task family** is `task_id: "harvest"` and success is checked from **inventory counts** (via `SuccessWrapper`), with a **time limit** (`TerminalWrapper`) and **death ends the episode** (`on_death=True` default).

#### 1) `harvest_log_in_plains`
- **Goal / success**: have **≥ 1 `log`** in inventory (`success_specs.any.item`).
- **World constraints**: forces `specified_biome: "plains"`.
- **Reward**: +1 when your inventory `log` count increases (delta-based).
- **Episode limit**: 1000 steps.
- **Training reset mode**: `fast_reset: 5`.

#### 2) `harvest_water_with_bucket`
- **Goal / success**: have **≥ 1 `water_bucket`** in inventory (`success_specs.all.item`).
- **Initial inventory**: spawns with **1 `bucket`** in mainhand.
- **World constraints**: `specified_biome: "plains"`.
- **Reward**: +1 on inventory delta for `water_bucket`.
- **Episode limit**: 1000 steps.
- **Training reset mode**: `fast_reset: 5`.

#### 3) `harvest_sand`
- **Goal / success**: have **≥ 1 `sand`** in inventory (`success_specs.all.item`).
- **World constraints**: none specified (biome left to MineDojo defaults).
- **Reward**: +1 on inventory delta for `sand`.
- **Episode limit**: 1000 steps.
- **Training reset mode**: `fast_reset: 5`.

#### 4) `shear_sheep`
- **Goal / success**: have **≥ 1 `wool`** in inventory (`success_specs.any.item`).
- **Initial inventory**: spawns with **1 `shears`** in mainhand.
- **Entity setup**: spawns `initial_mobs: "sheep"` within a box around the agent (from `[-15,1,-15]` to `[15,1,15]`).
- **World constraints**: `specified_biome: "plains"`.
- **Reward**: +1 on inventory delta for `wool`.
- **Episode limit**: 1000 steps.
- **Training reset mode**: `fast_reset: 5`.

#### 5) `mine_iron_ore`
- **Goal / success**: have **≥ 1 `iron_ore`** in inventory (`success_specs.all.item`).
- **Initial inventory**: spawns with **1 `stone_pickaxe`** in mainhand.
- **World constraints**: none specified (biome left to defaults).
- **Reward**: +1 on inventory delta for `iron_ore`.
- **Episode limit**: **2000** steps (longer horizon).
- **Training reset mode**: `fast_reset: 10`.

**Two important “speed knobs” used across tasks**
- **`break_speed_multiplier: 100`** is set in `sim_specs` for these tasks: breaking blocks is drastically sped up.
- **MineCLIP / LS-Imagine prompts** exist (`clip_specs.prompts`, `concentration_specs.prompts`) but **task success in this repo is still the inventory-based check above**.

### How the Minecraft world is generated (in this repo’s setup)
MineDojo ultimately runs a real Minecraft simulator and uses **Minecraft’s normal procedural world generation** (terrain/ores/structures) unless you override it. In *this repo*, the two key world-generation controls you actually use are:

#### 1) Biome constraints (sometimes)
Some tasks pass `specified_biome: "plains"` (log, water bucket, shear sheep). That means MineDojo will **spawn/reset into a plains biome world/region** for those tasks; tasks without it use MineDojo’s default biome/world sampling.

#### 2) Reset mode controls how often the world is “regenerated”
This repo deliberately mixes two reset behaviors:

- **Hard reset (`env.reset()`)**: MineDojo performs a full environment reset (this is the one that best matches “new world / fresh state” behavior).
- **Fast/semi-fast reset during training (`fast_reset: 5` or `10`)**: this repo wraps the simulator with a “semi-fast reset” that uses MineDojo’s `FastResetWrapper` most episodes, and only occasionally does a hard reset. Critically (repo comment): **fast reset “doesn’t replace blocks”**, so the world can accumulate changes across episodes; it also random-teleports within a range (here `random_teleport_range=200`) to vary spawn location without fully regenerating the world.

If you want, tell me whether you mean the “5 tasks” for **training** or **testing/rollout collection**—because your YAML uses `fast_reset: 0` for `collect_rollouts_*` and `test_*`, which changes world/regeneration behavior a lot.

