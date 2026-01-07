import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from PIL import Image, ImageDraw, ImageFont


TASK_RE = re.compile(r"(?:^|/|_)minedojo_(harvest_log_in_plains|harvest_water_with_bucket|harvest_sand|mine_iron_ore|shear_sheep)(?:/|$)")


@dataclass(frozen=True)
class RunPick:
    baseline: Optional["wandb.apis.public.Run"]
    lsimagine: Optional["wandb.apis.public.Run"]


def _extract_task(run: "wandb.apis.public.Run") -> Optional[str]:
    # Prefer config.task if present (e.g., "minedojo_harvest_sand")
    cfg = run.config or {}
    cfg_task = cfg.get("task") or cfg.get("task_name") or cfg.get("env_task") or cfg.get("env_id")
    if isinstance(cfg_task, str):
        if cfg_task.startswith("minedojo_"):
            t = cfg_task[len("minedojo_") :]
            if TASK_RE.search(f"/minedojo_{t}/"):
                return t
        # Some configs store "harvest_sand" without prefix.
        if TASK_RE.search(f"/minedojo_{cfg_task}/"):
            return cfg_task

    # Fall back to scanning string-valued config fields for a "minedojo_<task>" substring.
    try:
        for v in cfg.values():
            if isinstance(v, str):
                m = TASK_RE.search(v)
                if m:
                    return m.group(1)
    except Exception:
        pass

    # Fall back to parsing run.name which is set to logdir path in this repo.
    name = run.name or ""
    m = TASK_RE.search(name)
    if m:
        return m.group(1)
    return None


def _norm_tag(tag: str) -> str:
    # Make tag matching robust across "LS-Imagine" vs "ls-imagine" and "_" vs "-"
    return (tag or "").strip().lower().replace("_", "-")


def _has_tag(run: "wandb.apis.public.Run", tag: str) -> bool:
    want = _norm_tag(tag)
    if not want:
        return False
    tags = [_norm_tag(t) for t in (run.tags or [])]
    return want in tags


def _pick_latest(runs: Iterable["wandb.apis.public.Run"]) -> Optional["wandb.apis.public.Run"]:
    runs = [r for r in runs if r is not None]
    if not runs:
        return None
    # created_at is ISO string; pandas handles it well.
    return max(runs, key=lambda r: pd.to_datetime(getattr(r, "created_at", None) or "1970-01-01"))


def _fetch_history(
    run: "wandb.apis.public.Run",
    keys: List[str],
    max_samples: int = 20000,
) -> pd.DataFrame:
    # W&B returns a DataFrame with requested keys + _step, sometimes `step` too.
    df = run.history(keys=keys, samples=max_samples, pandas=True)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if "_step" in df.columns and "step" not in df.columns:
        df = df.rename(columns={"_step": "step"})
    return df


def _plot_one(
    task: str,
    baseline_run: Optional["wandb.apis.public.Run"],
    ls_run: Optional["wandb.apis.public.Run"],
    metric: str,
    outpath: Path,
    max_samples: int,
) -> None:
    # Square plot for easy collage comparisons.
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5), dpi=220)
    ax.set_title(f"{task} — {metric}")
    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    # Prefer square axes box if supported by matplotlib version.
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass

    def plot_series(run, label):
        if run is None:
            ax.plot([], [], label=f"{label} (missing)")
            return
        df = _fetch_history(run, keys=[metric], max_samples=max_samples)
        if df.empty or metric not in df.columns:
            ax.plot([], [], label=f"{label} (no history)")
            return
        df = df.dropna(subset=[metric])
        if df.empty:
            ax.plot([], [], label=f"{label} (all NaN)")
            return
        ax.plot(df["step"], df[metric], linewidth=1.6, label=label)

    plot_series(baseline_run, "DreamerV3 baseline")
    plot_series(ls_run, "LS-Imagine")
    ax.legend(loc="best", fontsize=8)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    # Some viewers/tools are picky about RGBA; rewrite as RGB to maximize compatibility.
    try:
        img = Image.open(outpath)
        if img.mode in ("RGBA", "LA"):
            rgb = Image.new("RGB", img.size, (255, 255, 255))
            rgb.paste(img, mask=img.split()[-1])
            rgb.save(outpath)
    except Exception:
        # Best-effort only; the original image may still be perfectly valid.
        pass


def _load_font(size: int = 24) -> ImageFont.ImageFont:
    # Best-effort: use default bitmap font if truetype isn't available.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _make_collage(
    task_to_img: Dict[str, Path],
    outpath: Path,
    grid: Tuple[int, int] = (2, 3),
    pad: int = 20,
    bg: Tuple[int, int, int] = (20, 20, 20),
) -> None:
    rows, cols = grid
    tasks = list(task_to_img.keys())
    imgs: List[Optional[Image.Image]] = []
    for t in tasks:
        try:
            imgs.append(Image.open(task_to_img[t]).convert("RGB"))
        except Exception:
            imgs.append(None)

    # Determine tile size by max width/height among available images.
    avail = [im for im in imgs if im is not None]
    if not avail:
        raise RuntimeError("No images available to build collage.")
    tile_w = max(im.width for im in avail)
    tile_h = max(im.height for im in avail)

    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = rows * tile_h + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg)
    draw = ImageDraw.Draw(canvas)
    font = _load_font(size=24)

    def paste_tile(idx: int, img: Optional[Image.Image], title: str):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (tile_h + pad)
        if img is None:
            # Empty slot
            draw.rectangle([x0, y0, x0 + tile_w, y0 + tile_h], outline=(80, 80, 80), width=3)
            draw.text((x0 + 10, y0 + 10), f"{title}\n(missing)", fill=(220, 220, 220), font=font)
            return
        # Fit image into tile without distortion (letterbox if needed)
        fitted = img.copy()
        fitted.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        px = x0 + (tile_w - fitted.width) // 2
        py = y0 + (tile_h - fitted.height) // 2
        canvas.paste(fitted, (px, py))
        # Title overlay
        draw.rectangle([x0, y0, x0 + tile_w, y0 + 40], fill=(0, 0, 0))
        draw.text((x0 + 10, y0 + 7), title, fill=(255, 255, 255), font=font)

    # Paste up to rows*cols tiles; tasks fill first 5 slots, remaining are blank.
    max_tiles = rows * cols
    for i in range(max_tiles):
        if i < len(tasks):
            paste_tile(i, imgs[i], tasks[i])
        else:
            paste_tile(i, None, "empty")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(outpath)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="agents-world-model-research")
    parser.add_argument("--project", default="LS-Imagine")
    parser.add_argument(
        "--tasks",
        default="harvest_log_in_plains,harvest_water_with_bucket,harvest_sand,mine_iron_ore,shear_sheep",
        help="Comma-separated task names without the 'minedojo_' prefix.",
    )
    parser.add_argument("--metric", default="train_success_rate", help="W&B scalar key to plot.")
    parser.add_argument("--baseline_tag", default="baseline", help="W&B run tag for baseline runs.")
    parser.add_argument("--lsimagine_tag", default="LS-Imagine", help="W&B run tag for LS-Imagine runs.")
    parser.add_argument(
        "--required_tags",
        default="successful_eval",
        help="Comma-separated list of tags that all selected runs must have. Use empty string to disable.",
    )
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parents[1] / "output_infos" / "wandb_plots"),
        help="Output directory (should be gitignored).",
    )
    args = parser.parse_args()

    # Ensure WANDB_API_KEY can be supplied via environment; script will still work if user is already logged in.
    if not os.environ.get("WANDB_API_KEY"):
        # No-op: wandb.Api() works if user has prior auth config in env/home.
        pass

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    required_tags = [t.strip() for t in (args.required_tags or "").split(",") if t.strip()]
    api = wandb.Api()
    runs = list(api.runs(f"{args.entity}/{args.project}"))

    by_task: Dict[str, Dict[str, List["wandb.apis.public.Run"]]] = {t: {"baseline": [], "ls": []} for t in tasks}
    for run in runs:
        # Filter runs primarily by tags first.
        if required_tags:
            tags = set(_norm_tag(t) for t in (run.tags or []))
            if not all(_norm_tag(t) in tags for t in required_tags):
                continue
        task = _extract_task(run)
        if task not in by_task:
            continue
        if _has_tag(run, args.baseline_tag):
            by_task[task]["baseline"].append(run)
        if _has_tag(run, args.lsimagine_tag):
            by_task[task]["ls"].append(run)

    picks: Dict[str, RunPick] = {}
    for task in tasks:
        picks[task] = RunPick(
            baseline=_pick_latest(by_task[task]["baseline"]),
            lsimagine=_pick_latest(by_task[task]["ls"]),
        )

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    task_to_img: Dict[str, Path] = {}
    for task, pick in picks.items():
        outpath = outdir / f"{task}.png"
        _plot_one(task, pick.baseline, pick.lsimagine, args.metric, outpath, args.max_samples)
        task_to_img[task] = outpath

    collage_path = outdir / "collage_2x3.png"
    _make_collage(task_to_img, collage_path, grid=(2, 3))
    print(f"Wrote per-task plots to: {outdir}")
    print(f"Wrote collage to: {collage_path}")


if __name__ == "__main__":
    main()


