from __future__ import annotations

from pathlib import Path

import pytest

from edgerunner.data.toy import make_toy_dataset


@pytest.fixture(scope="session")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """5 tiles of 1024² at 0.20 m (fold i = tile i, nodata strip on toy_4), eval windows 512."""
    return make_toy_dataset(tmp_path_factory.mktemp("toy"), n_tiles=5, size=1024, eval_size=512)
