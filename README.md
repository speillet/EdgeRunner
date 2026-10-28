# EdgeRunner

Entraînement de [SAM-Road](https://github.com/htcr/sam_road) (CVPRW 2024) sur les orthos IGN
BD ORTHO (COG 0.20 m), avec des labels BD TOPO dessinés à la volée. PyTorch Lightning + Hydra.
État et prochaines étapes : [ROADMAP.md](ROADMAP.md).

## En 4 commandes

```bash
uv tool install rust-just   # une fois (ou : uvx --from rust-just just <recette>)
just setup                  # submodule sam_road, env Python, checkpoint SAM ViT-B
just toy                    # jeu synthétique dans data/toy
just dev                    # 2 batchs train + val sur le toy
```

| Recette | Effet |
| --- | --- |
| `just test` | tests CPU + ruff |
| `just train data_root=/data [clé=valeur…]` | entraînement ; toute clé de `configs/train.yaml` se surcharge |
| `just smoke data_root=/data` | overlays PNG, stats des labels, débit (`outputs/smoke`) |
| `INDEX_DIR=… just cluster [train\|dev\|smoke]` | job SLURM 2 × 3 A40 (copie des données sur NVMe puis DDP) |

Exemples : `just train data_root=data/toy data.batch_size=1`, `just train gsd=0.8 patch_size=512`,
`just train clearml=true`, `just train data.labels.road_width_m=null` (largeur BD TOPO réelle).
Sur une carte 12 Go : 1024 px en batch 1 seulement (sinon `gsd=0.8 patch_size=512`).
Sorties : `outputs/<run_name>/<date>/` (TensorBoard `tb/`, checkpoints `ckpt/`).

## Données (`data_root`)

```text
data_root/
  index/tiles.parquet     tile_id, uri, fold, width, height, valid_frac, geometry (emprise L93)
  index/windows.parquet   fenêtres d'éval : tile_id, fold, row_off, col_off, size
  cog/<uri>               COG RVB uint8 0.20 m avec overviews
  bdtopo/troncon/         GeoParquet hive dept=XX : geometry WKB, struct bbox, attributs BD TOPO
```

Les vraies données viennent de `road_graph_extractor` (fichiers seulement) ; `just toy` produit
la même arborescence.

## Comment un exemple est fabriqué

1. Train : dalle tirée au prorata de `valid_frac`, fenêtre aléatoire (≥ 50 % de pixels valides),
   rot90. Val : fenêtres de `windows.parquet`, labels déterministes.
2. Lecture **de la seule fenêtre** dans le COG, à `gsd` (overview ×2 à 0.4 m).
3. Routes lues **une fois par emprise de dalle** (+ marge) et par worker. Filtres : en service,
   non fictives, ni sentier / escalier / bac, ni tunnel.
4. Graphe sur le patch agrandi de 64 m puis recadré (pas de fausse extrémité au bord). Masque route
   de 3 m, keypoints = nœuds de degré ≠ 2. Cibles topologiques : `sample_patch` de sam_road.

Distances en mètres dans la config, converties par `${px:m,${gsd}}` : changer `gsd` ou `patch_size`
garde la même géométrie au sol.

## Limites connues

- sam_road ne détecte pas un pont qui passe exactement sur un nœud.
- Offsets TopoNet en pixels bruts : les checkpoints cityscale (1 m/px) ne se transfèrent pas tels quels.
- Pas encore d'inférence sur dalle entière ni d'APLS/TOPO.
