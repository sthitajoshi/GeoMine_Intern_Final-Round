"""Small shared helpers: config, paths, logging, JSON."""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"


def load_config(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or ROOT / "configs" / "default.json").read_text())


class Logger:
    def __init__(self, name: str):
        (EXP / "logs").mkdir(parents=True, exist_ok=True)
        self.f = open(EXP / "logs" / f"{name}.log", "a", encoding="utf-8")
        self(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv)}")

    def __call__(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_default))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def environment() -> dict:
    import sklearn
    import scipy
    import torch
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "torch": torch.__version__, "cpu_threads": torch.get_num_threads()}
