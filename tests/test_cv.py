"""Synthèse de la validation croisée à partir des metrics.json des folds."""

import json
from pathlib import Path

import pytest

from edgerunner.cv import summarize


def test_summary_has_one_row_per_finished_fold_then_mean_and_std(tmp_path: Path) -> None:
    for k, iou in ((0, 0.2), (1, 0.4)):
        (tmp_path / f"fold{k}").mkdir()
        metrics = {"val_folds": [str(k)], "best_ckpt": "x.ckpt", "val_loss": 1.0, "road_iou": iou}
        (tmp_path / f"fold{k}" / "metrics.json").write_text(json.dumps(metrics))
    (tmp_path / "fold2").mkdir()  # still running: no metrics.json
    table = summarize(tmp_path)
    assert list(table.index) == ["fold0", "fold1", "mean", "std"]
    assert list(table.columns) == ["val_loss", "road_iou"]
    assert table.loc["mean", "road_iou"] == pytest.approx(0.3)
    assert table.loc["std", "val_loss"] == 0


def test_summary_without_finished_fold_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        summarize(tmp_path)
