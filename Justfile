# EdgeRunner. `just` lists the recipes; Hydra overrides go after the recipe name:
#   just train data_root=/data trainer.max_epochs=20
# Install just once: uv tool install rust-just

sam_ckpt := "checkpoints/sam_vit_b_01ec64.pth"

default:
    @just --list

# submodules (https), Python env, git hooks, SAM ViT-B checkpoint
setup: submodules
    uv sync --locked --extra clearml
    uv run pre-commit install
    mkdir -p checkpoints
    test -f {{sam_ckpt}} || curl -L --fail -o {{sam_ckpt}} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# synthetic dataset in data/toy (same layout as the real data)
toy:
    uv run python -m edgerunner.data.toy data/toy

# sam_road + its `sam` submodule (declared over SSH upstream: forced to https)
submodules:
    git submodule update --init third_party/sam_road
    git -C third_party/sam_road config submodule.sam.url https://github.com/htcr/segment-anything-road.git
    git -C third_party/sam_road submodule update --init sam

# unit tests + lint + format check + types (what the CI runs)
test:
    uv run pytest -q
    uv run ruff check src tests
    uv run ruff format --check src tests
    uv run pyright

# calculs en mémoire : pas de checkpoint, de dataset sur disque ni de modèle SAM
unit:
    uv run pytest -q tests/test_losses.py tests/test_sampling.py tests/test_transforms.py tests/test_batch.py tests/test_imports.py

# format and auto-fix the code
fmt:
    uv run ruff format src tests
    uv run ruff check --fix src tests

# 2 train + val batches on the toy set, 0.8 m / 512 px (fits a 12 GB GPU)
dev *args:
    uv run python -m edgerunner.train data_root=data/toy gsd=0.8 patch_size=512 trainer.fast_dev_run=2 {{args}}

# training (Hydra overrides, see configs/train.yaml)
train *args:
    uv run python -m edgerunner.train {{args}}

# overlays + label stats + throughput (outputs/smoke)
smoke *args:
    uv run python -m edgerunner.smoke {{args}}

# tiles.parquet + windows.parquet with road_graph_extractor's folds: just index <cog_dir> <out>
index *args:
    uv run python -m edgerunner.data.index {{args}}

# cross-validation table (mean, std over folds): just cv-report $DS/runs/samroad_<array job>
cv-report *args:
    uv run python -m edgerunner.cv {{args}}

# SLURM job on the internal cluster, mode = smoke | dev | train | cv (sbatch/<mode>.slurm)
cluster mode="train" *args:
    mkdir -p logs
    sbatch sbatch/{{mode}}.slurm {{args}}
