# EdgeRunner

Entraînement de [SAM-Road](https://github.com/htcr/sam_road) (CVPRW 2024) sur les orthophotos IGN
BD ORTHO (COG 0.20 m), avec des labels tirés de la BD TOPO **à la volée**. Le modèle prédit des
masques (routes, carrefours) et la topologie locale (quels points du réseau sont reliés), d'où
l'on reconstruit un graphe routier. PyTorch Lightning + Hydra.

État d'avancement et prochaines étapes : [ROADMAP.md](ROADMAP.md).

## Démarrage en 4 commandes

```bash
uv tool install rust-just   # une fois (ou : uvx --from rust-just just <recette>)
just setup                  # submodule sam_road, environnement Python, checkpoint SAM ViT-B
just toy                    # jeu synthétique dans data/toy
just dev                    # 2 batchs train + val sur le toy : vérifie que tout tourne
```

## L'idée en une phrase

sam_road fournit **le modèle** ; EdgeRunner fournit **tout ce qu'il faut pour le nourrir avec des
données IGN** sans rien pré-calculer, ainsi que l'outillage (config, entraînement, cluster).

Le dépôt amont travaille sur de petites images PNG chargées en mémoire, avec un graphe
pré-calculé par image. Nos dalles font 25 000 px de côté : EdgeRunner lit seulement la fenêtre
utile du COG et les routes de la dalle, et fabrique les labels de chaque patch au moment du tirage.

## Arborescence

```text
EdgeRunner/
├── Justfile                     [ER] une recette par action : setup, toy, test, dev, train, smoke, cluster
├── pyproject.toml / uv.lock     [ER] paquet `edgerunner` et dépendances figées (uv)
├── configs/
│   └── train.yaml               [ER] LA config (data_root, gsd, patch, data, model, trainer)
├── src/edgerunner/              [ER] tout le code propre au projet
│   ├── __init__.py                   px(m, gsd) et resolver ${px:…} : distances en mètres → pixels
│   ├── _sam_road.py                  pont vers third_party/sam_road (imports, noms uniques, wandb vide)
│   ├── data/
│   │   ├── roads.py                  lecture BD TOPO par emprise de dalle (GeoParquet, filtres)
│   │   ├── labels.py                 masques + graphe d'un patch, cibles topo via sam_road
│   │   ├── dataset.py                dataset à la volée, collate, make_loaders
│   │   └── toy.py                    générateur de données synthétiques au format réel
│   ├── models/
│   │   └── sam_road.py               SAMRoadModule : LightningModule hérité de SAMRoad
│   ├── train.py                      point d'entrée Hydra (loaders, modèle, Trainer, ClearML)
│   └── smoke.py                      overlays PNG, stats des labels, débit du DataLoader
├── tests/                       [ER] tests CPU sur le jeu synthétique (pytest)
├── sbatch/train_samroad.sh      [ER] job SLURM : copie des données sur NVMe puis DDP 2 × 3 A40
├── third_party/sam_road/        [SR] submodule htcr/sam_road, figé, jamais modifié
│   ├── model.py                      réseau SAMRoad, losses, métriques, optimiseur      → utilisé
│   ├── dataset.py                    GraphLabelGenerator (cibles topo) ; SatMapDataset  → en partie
│   ├── graph_utils.py                outils de graphe (igraph, croisements, NMS, BFS)   → utilisé
│   ├── sam/segment_anything/         encodeur ViT de SAM (fork des auteurs)             → utilisé
│   ├── inferencer.py, graph_extraction.py   inférence sur grande image             → plus tard (CP9)
│   ├── cityscale_metrics/, spacenet_metrics/  APLS / TOPO                          → plus tard (CP10)
│   └── train.py, config/, utils.py, cityscale/, spacenet/   pipeline amont         → non utilisé
├── README.md, ROADMAP.md        [ER] usage ; feuille de route pour reprendre le travail
│
│   non versionnés (.gitignore) :
├── checkpoints/                 poids SAM ViT-B de Meta (téléchargés par just setup)
├── data/                        données locales (data/toy)
├── outputs/                     runs et smoke tests
└── road_graph_extractor-main/   copie de référence de l'ancien projet : inspiration seulement
```

`[ER]` = écrit pour EdgeRunner ; `[SR]` = code amont de sam_road.

## Qui fait quoi : EdgeRunner ou sam_road

| Brique | Fournie par | Détail |
| --- | --- | --- |
| Réseau (encodeur SAM ViT-B, décodeur de masques, TopoNet) | sam_road `model.py` | utilisé tel quel |
| Poids pré-entraînés de l'encodeur | Meta (SAM ViT-B) | `checkpoints/sam_vit_b_01ec64.pth` |
| Losses, métriques (IoU, F1 topo), groupes de LR | sam_road `model.py` | utilisés tels quels |
| Boucle d'entraînement Lightning | **EdgeRunner** `models/sam_road.py` | sous-classe : batch uint8 → float, loss topo sûre sans route, images TensorBoard au lieu de wandb, paliers de LR configurables |
| Cibles topologiques (points, paires voisines, connectées ou non) | sam_road `GraphLabelGenerator.sample_patch` | **EdgeRunner** réécrit seulement son `__init__` (`WindowGraphLabeler`) : graphe local au patch, rayons en mètres |
| Outils de graphe (igraph, croisements, subdivision, NMS, BFS) | sam_road `graph_utils.py` | utilisés tels quels |
| Lecture des images (fenêtre COG à la GSD voulue) | **EdgeRunner** `data/dataset.py` | sam_road charge des PNG entiers en RAM |
| Lecture des routes (BD TOPO, filtres) | **EdgeRunner** `data/roads.py` | sam_road lit un graphe pickle pré-calculé |
| Masques route / keypoints et graphe du patch | **EdgeRunner** `data/labels.py` | sam_road lit des PNG pré-calculés |
| Tirage des patches, folds, fenêtres d'éval | **EdgeRunner** `data/dataset.py` | index `tiles.parquet` / `windows.parquet` |
| Config, entrée, logs, cluster | **EdgeRunner** | Hydra + Justfile + TensorBoard/ClearML + SLURM (sam_road : argparse + YAML + wandb) |
| Index, COG, GeoParquet BD TOPO | road_graph_extractor | fichiers seulement, pas de code |

Comment le code amont est branché : [_sam_road.py](src/edgerunner/_sam_road.py) ajoute
`third_party/sam_road` (et `sam/`) au `sys.path` et charge `model.py` / `dataset.py` sous les
noms `sam_road_model` / `sam_road_dataset`. Toute adaptation se fait par **sous-classe** dans
`src/edgerunner/` : le submodule reste identique à l'amont, ce qui permet de le mettre à jour en
changeant simplement de commit.

## Parcours d'un exemple d'entraînement

```text
index/tiles.parquet ──► tirage d'une dalle (au prorata de valid_frac) et d'une fenêtre
                         │  (refusée si < 50 % de pixels valides, lu sur overview)
                         ▼
cog/<dalle>.tif ───────► lecture de la seule fenêtre, rééchantillonnée à gsd   [dataset.py]
bdtopo/troncon/ ───────► routes de l'emprise de la dalle, en cache par worker  [roads.py]
                         ▼
                         render_window : patch agrandi de 64 m → masque route 3 m,
                         keypoints (degré ≠ 2), graphe {(row, col): voisins}    [labels.py]
                         ▼
                         WindowGraphLabeler.sample_patch : points + paires     [labels.py → sam_road]
                         ▼
                         rot90 aléatoire, collate → batch                       [dataset.py]
                         ▼
                         SAMRoadModule : SAMRoad.forward, loss masques + topo  [models → sam_road]
```

Pourquoi la marge de 64 m : une route qui sort du patch garde ses vrais voisins, donc pas de
fausse extrémité au bord, et le BFS de sam_road peut sortir du patch puis y revenir.

En validation : fenêtres fixes de `windows.parquet`, découpées en patches, sans rotation, avec des
labels identiques d'une époque à l'autre.

## Configuration

Un seul fichier, [configs/train.yaml](configs/train.yaml). Toute clé se surcharge après la recette :
`just train data.batch_size=8 trainer.max_epochs=20`.

| Clé | Défaut | Rôle |
| --- | --- | --- |
| `data_root` | `data` | dossier des données (arborescence ci-dessous) |
| `gsd` / `patch_size` | 0.4 / 1024 | résolution vue par le modèle et taille du patch (410 m de côté) |
| `neighbor_radius_m` | 64 | rayon du voisinage topologique, et marge de dessin des labels |
| `data.train_folds` / `data.val_folds` | 0–3 / 4 | folds de `tiles.parquet` utilisés |
| `data.labels.road_width_m` | 3.0 | largeur du masque route ; `null` = largeur BD TOPO |
| `model.*` | clés amont | config SAM-Road (LR, nombre d'échantillons topo…) ; rayons en `${px:…}` |
| `trainer.*` | 1 GPU, 16-mixed | arguments de `lightning.Trainer` (`fast_dev_run`, `strategy`…) |
| `clearml` | false | `true` : tâche ClearML qui reprend les logs TensorBoard |

Les distances sont écrites en mètres et converties par `${px:m,${gsd}}`. Passer de
`gsd=0.4 patch_size=1024` à `gsd=0.8 patch_size=512` garde la même géométrie au sol.

## Recettes

| Recette | Effet |
| --- | --- |
| `just setup` | submodule (https), `uv sync`, checkpoint SAM ViT-B |
| `just toy` | jeu synthétique dans `data/toy` |
| `just test` | tests CPU + ruff |
| `just dev [clé=valeur…]` | 2 batchs sur le toy à 0.8 m / 512 px (tient sur 12 Go) |
| `just train data_root=/data [clé=valeur…]` | entraînement |
| `just smoke data_root=/data` | overlays PNG, stats des labels, débit (`outputs/smoke`) |
| `INDEX_DIR=… just cluster [train\|dev\|smoke]` | job SLURM (voir l'en-tête de `sbatch/train_samroad.sh`) |

Sur une carte de 12 Go, 1024 px ne passe qu'en batch 1 (sinon `gsd=0.8 patch_size=512`).

## Données attendues (`data_root`)

```text
data_root/
  index/tiles.parquet     tile_id, uri, fold, width, height, valid_frac, geometry (emprise L93)
  index/windows.parquet   fenêtres d'éval : tile_id, fold, row_off, col_off, size
  cog/<uri>               COG RVB uint8 0.20 m avec overviews
  bdtopo/troncon/         GeoParquet hive dept=XX : geometry WKB, struct bbox, attributs BD TOPO
```

Les vraies données sont produites par road_graph_extractor (seuls les fichiers sont utilisés).
`just toy` génère la même arborescence avec des dalles factices.

## Sorties d'un run

```text
outputs/<run_name>/<date>/
  .hydra/config.yaml      config complète du run, overrides inclus (interpolations ${…} non résolues)
  train.log               log texte
  tb/                     TensorBoard : losses, IoU, F1 topo, LR, images val/masks ;
                          hparams.yaml = config SAM-Road résolue (rayons en pixels)
  ckpt/                   3 meilleurs checkpoints (val_loss) + last.ckpt
```

`uv run tensorboard --logdir outputs` pour suivre les courbes.

## Tests

| Fichier | Ce qui est vérifié |
| --- | --- |
| `tests/test_labels.py` | carrefours et extrémités, rien au bord ni sur un pont, largeur du masque, rotation |
| `tests/test_dataset.py` | filtres BD TOPO, format du batch, grille d'éval, GSD 0.4 vs 0.8, workers |
| `tests/test_model.py` | losses finies, loss topo à 0 sans route, paliers de LR (CPU, faux checkpoint) |
| `tests/test_config.py` | conversion mètres → pixels, loaders construits depuis la config |

## Limites connues

- sam_road ne détecte pas un pont qui passe exactement sur un nœud.
- Offsets TopoNet en pixels bruts : les checkpoints cityscale (1 m/px) ne se transfèrent pas tels quels.
- Pas encore d'inférence sur dalle entière ni d'APLS/TOPO (ROADMAP, CP9 et CP10).

## Licences et citation

| Élément | Licence |
| --- | --- |
| EdgeRunner (ce dépôt hors `third_party/`) | MIT ([LICENSE](LICENSE)) |
| `third_party/sam_road` (htcr) | MIT |
| `third_party/sam_road/sam` (fork de Segment Anything) | Apache-2.0 |
| Poids SAM ViT-B (`sam_vit_b_01ec64.pth`, Meta) | Apache-2.0 |

Pour citer : [CITATION.cff](CITATION.cff) (bouton « Cite this repository » sur GitHub), qui
référence aussi les deux papiers sur lesquels EdgeRunner repose :

```bibtex
@article{hetang2024segment,
  title={Segment Anything Model for Road Network Graph Extraction},
  author={Hetang, Congrui and Xue, Haoru and Le, Cindy and Yue, Tianwei and Wang, Wenping and He, Yihui},
  journal={arXiv preprint arXiv:2403.16051},
  year={2024}
}
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
