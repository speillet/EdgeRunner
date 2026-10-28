"""Access to the htcr/sam_road submodule, which is a folder of flat modules, not a package.

`third_party/sam_road` goes on sys.path for its own imports (`graph_utils`, `sam.segment_anything`),
plus `third_party/sam_road/sam` because `segment_anything/predictor.py` imports `segment_anything`
absolutely. `model.py` and `dataset.py` are loaded under unique names so they cannot clash with
other `model` / `dataset` modules.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

SAM_ROAD_DIR = Path(
    os.environ.get("SAM_ROAD_DIR", Path(__file__).resolve().parents[2] / "third_party" / "sam_road")
)


def _setup() -> None:
    if not (SAM_ROAD_DIR / "sam" / "segment_anything").is_dir():
        raise ImportError(
            f"sam_road submodule missing or incomplete in {SAM_ROAD_DIR}. Run:\n"
            "  git submodule update --init third_party/sam_road\n"
            "  git -C third_party/sam_road config submodule.sam.url "
            "https://github.com/htcr/segment-anything-road.git\n"
            "  git -C third_party/sam_road submodule update --init sam"
        )
    for p in (SAM_ROAD_DIR, SAM_ROAD_DIR / "sam"):
        if str(p) not in sys.path:
            sys.path.append(str(p))  # appended: never shadows installed packages


def _load(name: str, filename: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SAM_ROAD_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_setup()
graph_utils = importlib.import_module("graph_utils")
sam_dataset = _load("sam_road_dataset", "dataset.py")


def sam_model() -> ModuleType:
    """`model.py` (torchvision, segment_anything): only loaded where needed."""
    if importlib.util.find_spec("wandb") is None:  # imported by model.py, only its viz uses it
        sys.modules["wandb"] = ModuleType("wandb")
    return _load("sam_road_model", "model.py")
