"""Les briques réutilisables doivent fonctionner sans le sous-module SAM-Road."""

import os
import subprocess
import sys


def test_core_modules_import_without_sam_or_lightning() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import edgerunner.models.losses
import edgerunner.data.batch
import edgerunner.data.sampling
import edgerunner.data.transforms
import edgerunner.data.raster
assert 'sam_road_model' not in sys.modules
assert 'sam_road_dataset' not in sys.modules
assert 'lightning' not in sys.modules
""",
        ],
        env={**os.environ, "SAM_ROAD_DIR": "/nonexistent/edgerunner-test-sam"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
