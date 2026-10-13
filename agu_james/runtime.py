"""Explicit artifact paths and parameter overrides for retained source modules."""
from contextlib import contextmanager
from pathlib import Path
import json
import os

_CONFIG = {}

def configure(config=None):
    global _CONFIG
    _CONFIG = dict(config or {})
    return _CONFIG

def artifact_root():
    return Path(_CONFIG.get("artifact_root", "artifacts")).resolve()

def path(key, default=None):
    value = _CONFIG.get("paths", {}).get(key, default)
    if value is None:
        if key.startswith('agu_james.'):
            value = artifact_root() / 'inputs' / key.rsplit('.', 1)[-1]
        else:
            raise ValueError(f"Set paths.{key} in the config file")
    return Path(value).expanduser().resolve()

def setting(key, default=None):
    if key in _CONFIG.get("overrides", {}):
        value = _CONFIG["overrides"][key]
        name = key.rsplit('.', 1)[-1]
        if isinstance(value, str) and (name.endswith(('_PATH','_DIR','_ROOT','_NPZ','_path','_dir','_root','_folder')) or name in {'BASE','ROOT','HERE','DATA','BEST_TRACK','CSV_PATH','OUTPUT_CSV','FEATURE_TABLE','FEATURE_CSV'}):
            return Path(value).expanduser().resolve()
        return value
    return default() if callable(default) else default

def legacy_path(relative):
    """Known training assets live under the configured artifact root."""
    rel = str(relative).replace("\\", "/")
    aliases = {
        "../../era5/lat.npy": "data/lat.npy", "../../era5/lon.npy": "data/lon.npy",
        "../../npz/prs_variable_stats.csv": "statistics/prs_variable_stats.csv",
        "../../npz/sfc_regular_variable_stats.csv": "statistics/sfc_regular_variable_stats.csv",
        "../../npz/sfc_average_variable_stats.csv": "statistics/sfc_average_variable_stats.csv",
        "target_mean_std.csv": "statistics/target_mean_std.csv",
        "next_feature_table_weight.csv": "statistics/next_feature_table_weight.csv",
        "next_feature_table_weight_ver3.csv": "statistics/next_feature_table_weight_ver3.csv",
    }
    if rel not in aliases:
        raise ValueError(f"Unmapped legacy training asset: {relative}")
    if rel in {'../../era5/lat.npy','../../era5/lon.npy'}:
        key='lat' if rel.endswith('/lat.npy') else 'lon'
        if key in _CONFIG.get('paths',{}):return path(key)
    return path(aliases[rel], artifact_root() / aliases[rel])

@contextmanager
def workdir(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(directory)
    try:
        yield directory
    finally:
        os.chdir(old)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
