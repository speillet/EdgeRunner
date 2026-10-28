"""EdgeRunner: SAM-Road on IGN BD ORTHO. Importing it registers the `${px:metres,gsd}` resolver."""

from omegaconf import OmegaConf


def px(metres: float, gsd: float) -> int:
    """Ground distance (m) → pixels at `gsd` (m/px), at least 1."""
    return max(1, round(float(metres) / float(gsd)))


if not OmegaConf.has_resolver("px"):
    OmegaConf.register_new_resolver("px", px)
