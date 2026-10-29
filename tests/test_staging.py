"""Exercise the actual NVMe staging shell block without SLURM or real transfers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("fail_copy", [False, True])
def test_staging_propagates_background_copy_failure(tmp_path: Path, fail_copy: bool) -> None:
    script = (Path(__file__).resolve().parents[1] / "sbatch/train_samroad.sh").read_text()
    block = script.split("bash -c '\n", 1)[1].split("'\n", 1)[0]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    rsync = fake_bin / "rsync"
    rsync.write_text(
        '#!/bin/bash\nfor arg in "$@"; do\n'
        '  if [[ "$arg" == --files-from=* ]]; then\n'
        '    cat "${arg#--files-from=}" >> "$COPIED"\n'
        '    if [ "$FAIL_COPY" = 1 ] && [[ "$arg" == *_aa ]]; then exit 23; fi\n'
        "  fi\ndone\n"
    )
    rsync.chmod(0o755)
    tiles = tmp_path / "tiles.txt"
    tiles.write_text("".join(f"tile_{i}.tif\n" for i in range(16)))
    copied = tmp_path / "copied.txt"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DS": str(tmp_path / "store"),
        "NVME": str(tmp_path / "nvme"),
        "INDEX_DIR": str(tmp_path / "index"),
        "CKPT": "sam.pth",
        "TILES": str(tiles),
        "SLURM_JOB_ID": f"edgerunner_test_{os.getpid()}_{int(fail_copy)}",
        "COPIED": str(copied),
        "FAIL_COPY": str(int(fail_copy)),
    }
    result = subprocess.run(["bash", "-c", block], env=env, capture_output=True, text=True)
    assert (result.returncode != 0) == fail_copy, result.stderr
    assert sorted(copied.read_text().splitlines()) == sorted(tiles.read_text().splitlines())
    assert ("COG staging failed" in result.stderr) == fail_copy
    assert not list(Path("/tmp").glob(f"samroad_{env['SLURM_JOB_ID']}_*"))
