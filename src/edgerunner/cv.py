"""Cross-validation summary: one row per fold (its best checkpoint on its val fold), mean, std.

    just cv-report $DS/runs/samroad_<array job>     (also run by each fold at the end of cv.slurm)
Reads <dir>/fold*/metrics.json (written by edgerunner.train), writes <dir>/cv_summary.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(cv_dir: str | Path) -> pd.DataFrame:
    rows = {}
    for path in sorted(Path(cv_dir).glob("fold*/metrics.json")):
        metrics = json.loads(path.read_text())
        rows[path.parent.name] = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, int | float) and not isinstance(v, bool)
        }
    if not rows:
        raise FileNotFoundError(f"no fold*/metrics.json in {cv_dir}")
    table = pd.DataFrame.from_dict(rows, orient="index")
    return pd.concat([table, pd.DataFrame([table.mean(), table.std()], index=["mean", "std"])])


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-validation summary.")
    p.add_argument("cv_dir", help="folder holding fold<k>/metrics.json")
    a = p.parse_args()
    table = summarize(a.cv_dir)
    out = Path(a.cv_dir) / "cv_summary.csv"
    tmp = out.with_suffix(".tmp")
    table.to_csv(tmp)
    tmp.replace(out)  # two folds may finish together
    print(table.round(4).to_string(), f"\n{len(table) - 2} folds done → {out}")


if __name__ == "__main__":
    main()
