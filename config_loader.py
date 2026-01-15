import pathlib
from typing import Optional, List
import ruamel.yaml as yaml


def deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge `update` into `base` (mutates and returns base)."""
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_file(path: pathlib.Path) -> dict:
    text = path.read_text()
    data = yaml.safe_load(text) if text.strip() else {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping at top-level: {path}")
    return data


def parse_set_kv(expr: str):
    if "=" not in expr:
        raise ValueError(f"--set expects key=value but got: {expr}")
    key, raw = expr.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    value = yaml.safe_load(raw) if raw != "" else ""
    return key, value


def set_by_dotted_key(cfg: dict, dotted_key: str, value):
    parts = dotted_key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def resolve_mode_files(mode: str, configs_dir: pathlib.Path) -> list[pathlib.Path]:
    base = configs_dir / "base.yaml"
    env = configs_dir / "env" / "minedojo.yaml"
    mode_dir = configs_dir / "mode"

    # Friendly mapping from historical mode names to mode files.
    mode_map = {
        "dreamer_baseline": "dreamer_pixels_baseline.yaml",
        "dreamer_pixels_baseline": "dreamer_pixels_baseline.yaml",
        "ls_imagine": "ls_imagine.yaml",
        "mpf_lsd": "mpf_lsd.yaml",
        "mpf_lsd_baseline": "mpf_lsd_baseline.yaml",
        "vlm_only_baseline": "vlm_only_baseline.yaml",
        "vlm_only_baseline_lora": "vlm_only_baseline_lora.yaml",
    }
    fname = mode_map.get(mode, f"{mode}.yaml")
    mode_path = mode_dir / fname
    return [base, env, mode_path]


def load_config_paths(paths: List[pathlib.Path], local_path: Optional[pathlib.Path] = None) -> dict:
    cfg: dict = {}
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(str(p))
        deep_merge(cfg, load_yaml_file(p))
    if local_path is not None and local_path.exists():
        deep_merge(cfg, load_yaml_file(local_path))
    return cfg

