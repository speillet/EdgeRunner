# ROADMAP — EdgeRunner (SAM-Road sur BD ORTHO IGN)

> **Pour l'agent qui reprend (Codex, Claude, …)** : lire ce fichier en entier avant de coder.
> À la fin de chaque checkpoint : cocher les cases, remplir le **Journal** (§7), mettre à jour
> « Reprendre ici » (§0) et le `README.md` si l'usage change.

## 0. Reprendre ici

- **Dernière mise à jour** : 2026-10-29.
- **Fait** : CP0 → CP6, CP6b (simplification : 1 YAML, Justfile, plus de wandb) et CP6c
  (bonnes pratiques : CI, pre-commit, pyright, provenance, torch cu126, licence). Validé
  **uniquement sur données synthétiques**, en local.
- **Revue du 2026-10-29** : corrections nodata, cache raster et copies NVMe ; 36 tests CPU
  réussis, Ruff/formatage/Pyright et syntaxe Bash validés. Exécution SLURM à vérifier.
- **Modularisation du 2026-10-29** : calcul des losses, lecture raster, choix des fenêtres,
  visibilité et collate isolés ; contrats typés ; script NVMe autonome ; RNG NumPy restauré.
- **Cluster interne du 2026-10-29** (CP6d) : un `.slurm` par job, 1 nœud × 3 A40, index
  reconstruit avec les folds de road_graph_extractor (train 1–4, val 0, `test` exclu).
  Validation croisée 5 folds (CP6e) : `just cluster cv`, synthèse `cv_summary.csv`.
  Validation : 52 tests dont 12 via `just unit` ; qualité du code et syntaxe Bash validées.
- **Prochain** : **CP7** (validation sur vraies dalles, accès au cluster SLURM requis).
  Sans accès au cluster : faire **CP9** (inférence sur dalle entière) sur le jeu toy.
- **Git** : branche `main` sur github.com/speillet/EdgeRunner, un commit par élément du dépôt
  (`git log --reverse --oneline` se lit comme une visite guidée). Nouveaux commits : seulement
  avec l'autorisation explicite de l'utilisateur.

Vérifier l'état en 2 minutes :

```bash
uv tool install rust-just    # ou préfixer chaque commande par : uvx --from rust-just
git submodule status         # sam_road 04b67c5 ; sam 6fdee8f
just setup                   # idempotent
just test                    # tests CPU, ruff, formatage et pyright
just toy && just dev         # exit 0
```

## 1. Règles du projet (non négociables)

1. `road_graph_extractor-main/` (gitignoré) sert **d'inspiration uniquement** : ni import, ni
   dépendance, ni copier-coller. On consomme seulement les **fichiers de données** qu'il a produits
   sur le cluster (index parquet, COG, GeoParquet BD TOPO).
2. Le code amont `third_party/sam_road` n'est **jamais modifié** : sous-classes ou réécriture
   ciblée dans `src/edgerunner/`, accès par le shim `edgerunner/_sam_road.py`.
3. Aucun commit ni push sans autorisation explicite de l'utilisateur.
4. Fichiers < 700 lignes (sinon découper). Ruff (ligne 100) + `ruff format` + pyright standard.
   Tests CPU dans `tests/`. `just test` doit passer avant tout commit : c'est ce que rejoue la CI,
   et les hooks pre-commit formatent / vérifient chaque commit.
5. Docs en français, concises. Réponses courtes à l'utilisateur ; débattre les choix structurants
   avec lui **avant** d'implémenter.
6. Cluster : aucun calcul sur le nœud de login, tout passe par `sbatch`.
7. Décisions déjà tranchées avec l'utilisateur, à ne pas rouvrir sans lui :

| Sujet | Décision |
| --- | --- |
| Résolution | GSD 0.40 m, patch 1024 px (= une fenêtre d'éval de 2048 px natifs) |
| Distances | En mètres dans Hydra, converties par `${px:m,${gsd}}` |
| Usage | Un seul `configs/train.yaml` (pas de groupes Hydra), un `data_root` à arborescence fixe, recettes `just` |
| « Emprise de la dalle » | Lire seulement la fenêtre COG + les routes de l'emprise de la dalle (+ marge) |
| Intégration sam_road | Submodule + sous-classe |
| Loggers | TensorBoard (défaut) + ClearML (`clearml=true`, rang 0, capture TB). Pas wandb (module factice) |
| Masque route | Étroit, 3 m fixe (largeur BD TOPO en option) ; keypoints = degré ≠ 2 |
| Crops sans route | Gardés ; dénominateur de la loss topo clampé à 1 |

## 2. Carte du code

| Fichier | Rôle |
| --- | --- |
| `src/edgerunner/__init__.py` | `px` (m → px) et resolver OmegaConf `${px:…}`, enregistré à l'import |
| `src/edgerunner/_sam_road.py` | Met `third_party/sam_road` (+ `/sam`) sur le `sys.path`, charge `dataset.py` / `model.py` sous des noms uniques, module `wandb` factice |
| `src/edgerunner/data/roads.py` | BD TOPO par bbox via `pyarrow.dataset` (filtres : en service, non fictif, ni sentier / escalier / bac, ni tunnel) |
| `src/edgerunner/data/labels.py` | `render_window` (masques + graphe sat2graph sur patch + marge), `WindowGraphLabeler` (`__init__` amont paramétré), `topo_labels` |
| `src/edgerunner/data/dataset.py` | Orchestration `IGNSamRoadDataset` et `make_loaders` ; réexport de `collate` |
| `src/edgerunner/data/raster.py` | Lecture de fenêtres et cache LRU des handles par processus |
| `src/edgerunner/data/sampling.py` | Grille d’éval, tirage des fenêtres et contexte RNG NumPy amont |
| `src/edgerunner/data/types.py`, `batch.py`, `transforms.py` | Contrats typés, collate et visibilité des paires |
| `src/edgerunner/models/losses.py` | Losses sur tenseurs, sans SAM ni Lightning |
| `src/edgerunner/data/toy.py` | Jeu synthétique au même format que les vraies données |
| `src/edgerunner/data/index.py` | `tiles.parquet` / `windows.parquet` des COG, folds identiques à ceux de road_graph_extractor (zones, graine 0) |
| `src/edgerunner/models/sam_road.py` | `SAMRoadModule(SAMRoad)` |
| `src/edgerunner/train.py` | `@hydra.main` : construction explicite (pas de `_target_`), `build_loaders(cfg)`, `provenance(data_root)`, ClearML optionnel |
| `src/edgerunner/smoke.py` | Overlays PNG, stats des labels, débit du DataLoader |
| `src/edgerunner/cv.py` | Synthèse de la validation croisée : `fold*/metrics.json` → `cv_summary.csv` (moyenne, écart-type) |
| `configs/train.yaml` | **La** config : `data_root`, `gsd`, `patch_size`, sections `data`, `model` (clés amont), `trainer` |
| `Justfile` | `setup`, `submodules`, `toy`, `test`, `fmt`, `dev`, `train`, `smoke`, `index`, `cv-report`, `cluster [smoke\|dev\|train\|cv]` |
| `.github/workflows/ci.yml` | CI : `just submodules`, `uv sync --locked --extra clearml`, `just test`, hooks pre-commit |
| `.pre-commit-config.yaml` | hooks : YAML/TOML, fin de fichier, espaces, gros fichiers, ruff check + format (version de uv.lock) |
| `LICENSE`, `CITATION.cff` | MIT ; citation du dépôt + SAM-Road + Segment Anything |
| `sbatch/{smoke,dev,train,cv}.slurm`, `common.sh`, `stage_data.sh` | Un job SLURM par fichier ; partie commune (env, index, copie) ; copie NVMe testable séparément |

## 3. Contrats à ne pas casser

- **Batch** (`collate`) : `rgb` uint8 [B,P,P,3] · `keypoint_mask`, `road_mask` uint8 {0,1} [B,P,P]
  (le module les convertit en float dans `on_after_batch_transfer`) · `graph_points` float32
  [B,N,2] en (x, y) dans le repère du patch, complété par des zéros · `pairs` int32 [B,S,Q,2] ·
  `connected`, `valid` bool [B,S,Q] · `pixel_valid` bool [B,P,P] (image disponible) · `tile_id`, `row_off`, `col_off`, `rot` en listes.
- **`data_root`** : `index/{tiles,windows}.parquet`, `cog/<uri>`, `bdtopo/troncon/dept=XX/`.
- **Index** : `tiles.parquet` (`tile_id`, `uri` relatif à `data_root/cog`, `fold`, `width`, `height`,
  `valid_frac`, `geometry` L93) ; `windows.parquet` (`tile_id`, `fold`, `row_off`, `col_off`,
  `size`). Les folds sont des chaînes (`"0"`…`"4"`, `test`, `ext_test`).
- **BD TOPO** : un dossier hive `dept=XX`, `geometry` en WKB, colonne struct `bbox`
  {xmin, ymin, xmax, ymax}, attributs `nature`, `largeur_de_chaussee`, `etat_de_l_objet`,
  `fictif`, `position_par_rapport_au_sol`.
- **Rotation** : image et masques avec `np.rot90(a, rot, (0, 1))` ; les points sont tournés par
  `sample_patch(…, rot_index=rot)` en amont (testé dans `test_points_follow_rotation`).

## 4. Checkpoints faits

- [x] **CP0 — Socle** : `pyproject.toml` (uv, hatchling), `.gitignore`, submodule sam_road
  (URL https forcée pour le sous-module `sam`). *Preuve* : `git submodule status`.
- [x] **CP1 — Shim sam_road + resolver** (`_sam_road.py`, resolver aujourd'hui dans `__init__.py`).
- [x] **CP2 — Labels** (`roads.py`, `labels.py`). *Preuve* : `uv run pytest tests/test_labels.py`.
- [x] **CP3 — Dataset + loaders + jeu toy**. *Preuve* : `uv run pytest tests/test_dataset.py`
  (format du batch, filtres BD TOPO, grille d'éval, GSD 0.4 vs 0.8, pickle, workers fork /
  forkserver).
- [x] **CP4 — LightningModule**. *Preuve* : `uv run pytest tests/test_model.py` (checkpoint SAM
  factice, loss topo = 0 sur un batch sans route).
- [x] **CP5 — Hydra + entraînement local**. *Preuve* : `fast_dev_run` passe ; mini-run à 1024 px,
  batch 1, loss 1.47 → 0.35 en 30 pas ; `outputs/<run>/ckpt/*.ckpt` et TensorBoard écrits
  (scalaires + image `val/masks`).
- [x] **CP6 — Outils cluster + docs** : `sbatch/train_samroad.sh` (syntaxe vérifiée par
  `bash -n`, **jamais lancé**), smoke, `README.md`.
- [x] **CP6b — Simplification** (demande de l'utilisateur : code minimaliste, facile à utiliser) :
  11 YAML → 1, DataModule → `make_loaders`, `instantiate` → construction explicite, `data_root`
  unique, Justfile, smoke dans le package, mode `smoke` du sbatch, wandb retiré.
  *Preuve* : `just test` (25 passed), `just toy && just dev`, `just smoke data_root=data/toy`.
- [x] **CP6c — Bonnes pratiques** : `.env` / `clearml.conf` ignorés ; `.python-version` 3.12 et
  `uv sync --locked` ; torch cu126 (driver des A40 non vérifié pour CUDA 13) ; pre-commit +
  `ruff format` ; pyright 0 erreur ; provenance des runs dans les hparams ; licence MIT +
  CITATION.cff ; CI GitHub Actions. *Preuve* : `just test` (26 passed), badge CI vert, checkpoint
  rechargé avec `load_from_checkpoint` et sa provenance.
- [x] **CP6d — Jobs du cluster interne** : `sbatch/{smoke,dev,train}.slurm` (1 nœud × 3 A40,
  `DS=/mnt/stores/store-DAI/…`), `sbatch/common.sh`, `edgerunner.data.index` (folds de
  road_graph_extractor reconstruits), défauts `train_folds` 1–4 / `val_folds` 0.
  *Preuve* : `just test` (54 passed) ; index = 259 dalles, 54 zones, 37 296 fenêtres comme sa
  doc, les 36 dalles de `$DS/runs/1357/eval_fold0` en fold 0 ; `train.slurm` et `smoke.slurm`
  simulés hors SLURM (faux `srun`). **Jamais soumis au vrai SLURM.**
- [x] **CP6e — Validation croisée** : `sbatch/cv.slurm` (array 0-4%2, tâche k = val sur le fold k,
  train sur les 4 autres), `metrics.json` en fin d'entraînement (meilleur checkpoint réévalué),
  `edgerunner.cv` → `cv_summary.csv`. *Preuve* : `just test` (56 passed) ; 2 tâches de
  `cv.slurm` simulées sur le toy avec un vrai entraînement court (faux `srun`, copie NVMe
  sautée) : dossiers `fold0/`, `fold1/`, `metrics.json` et synthèse corrects.

## 5. Checkpoints à venir

### CP7 — Validation sur vraies données (cluster) ⟵ PROCHAIN
- **Prérequis** : accès SLURM. Store partagé
  `$DS=/mnt/stores/store_dai/pocs/datasets/road_graph_extractor` (chemin sur le serveur de
  l'utilisateur, défaut du sbatch) : `cog/` (259 COG), `bdtopo/bretagne/troncon/`,
  `bdtopo/annot_depts/troncon/`, `annotations/`, `runs/` (DeH4R), `checkpoints/` (SAM ViT-B déposé
  le 2026-10-28). Sur les nœuds du cluster : `DS=/mnt/stores/store-DAI/pocs/datasets/road_graph_extractor`.
  **Index officiel absent** de `$DS/index/` : les jobs le reconstruisent alors dans
  `$OUT/index` (`edgerunner.data.index`, mêmes folds), et prendront l'officiel dès sa copie.
- **Déjà validé en local (2026-10-28)** sur 6 vraies dalles copiées dans `data/ign/` (5 autour de
  Rennes en train, 1 à Saint-Brieuc en val) avec un **index de test provisoire** : lecture BD TOPO
  réelle OK, overlays alignés, 109 exemples/s avec 8 workers, entraînement de 3 époques × 600
  patches (0.4 m, 1024 px, batch 1 sur A2000) sans erreur, val_loss 0.092 → 0.071. Ces 6 dalles
  sont toutes dans le fold `test` de road_graph_extractor : chiffres de mise au point seulement.
- **Étapes** :
  1. Cloner EdgeRunner sur le cluster, initialiser le submodule (commandes du README).
  2. `just cluster smoke` (1 nœud : copie NVMe puis `edgerunner.smoke`,
     overlays dans `$OUT/smoke`).
  3. Vérifier que `read_roads` lit le GeoParquet réel (écrit par DuckDB 1.5) : type de la colonne
     `geometry` (binaire ou extension Parquet GEOMETRY, gérée par `_wkb`) et présence de `bbox`.
  4. Faire relire les overlays par l'utilisateur : masques sur les routes (décalage BD TOPO d'environ
     2.5 m attendu), keypoints aux carrefours, rien sur les ponts.
  5. `just cluster dev` (1 nœud × 3 A40), puis `just cluster train`.
- **Fini quand** : overlays validés ; débit en échantillons/s mesuré depuis le NVMe et ≥ besoin
  des GPU (3 × batch × it/s par nœud) ; taux de patchs sans route noté ; run DEV terminé avec le
  code 0 ; tâche ClearML visible. Chiffres reportés au §7.

### CP8 — Premier entraînement complet et réglages
- Batch par GPU sur A40 (essayer 4 puis 8), LR, `max_epochs`, `samples_per_epoch`, et
  `model.lr_milestones` (en **époques**, à aligner sur `max_epochs`).
- Suivre `val_loss`, `road_iou`, `keypoint_iou`, `topo_f1`.
- **Fini quand** : run complet sans NaN, courbes de validation qui descendent, meilleur
  checkpoint enregistré, job id et métriques au §7.
- **Piège** : si DDP signale des paramètres inutilisés (variantes `FREEZE_ENCODER`, LoRA,
  `TOPONET_VERSION`), passer `trainer.strategy=ddp_find_unused_parameters_true`.

### CP9 — Inférence sur dalle entière (faisable en local sur le toy)
- **But** : `src/edgerunner/infer.py` (+ recette `just infer`, et une section `infer:` dans
  `configs/train.yaml` plutôt qu'un 2e YAML) : checkpoint + dalle COG → graphe
  géoréférencé L93 (GeoParquet de LineStrings) et masques en option.
- **Ne pas importer** `third_party/sam_road/inferencer.py` : il appelle `parse_args()` à l'import
  et importe des modules aux noms génériques (`model`, `dataset`, `utils`). Réécrire la boucle de
  `infer_one_img` (lignes 61–235) dans `edgerunner`, en réutilisant `graph_utils` et
  `graph_extraction.extract_graph_points`. `graph_extraction.py` importe `tcod` et
  `scikit-image` : soit les ajouter aux dépendances, soit réécrire `extract_graph_points`
  (petit : seuil + `nms_points`).
- Découpage : blocs (ex. 4096 px à 0.4 m) avec recouvrement ≥ `NEIGHBOR_RADIUS` +
  `SAMPLE_MARGIN` ; dans chaque bloc, patches de 1024 se chevauchant (le pas
  `INFER_PATCHES_PER_EDGE` amont suppose une image de 2048 : le recalculer) ; fusion des masques
  par moyenne ; points, puis topologie ; couture des blocs (dédoublonner les nœuds dans les
  recouvrements).
- Chargement : `SAMRoadModule.load_from_checkpoint(ckpt, cfg={**hparams_cfg, "SAM_CKPT_PATH": …})`,
  car le `__init__` amont recharge les poids SAM.
- **Fini quand** : sur le toy, le graphe superposé à la dalle suit la grille peinte ; test unitaire
  de la couture ; temps par dalle mesuré.

### CP10 — Métriques APLS / TOPO
- Deux échelles (décision du projet d'origine) : fenêtres de 2048 px du fold `test` pour les
  ablations ; dalle entière recousue pour les jalons.
- GT : graphe BD TOPO (mêmes filtres que `read_roads`) ; test externe = annotations
  `$DS/annotations` (polygones + `route_graph_skel` / `route_graph_snap`), fichiers seulement.
- Outils : `third_party/sam_road/*_metrics` (APLS en Go, TOPO en Python) ou APLS en Python pur.
  **Choix à débattre avec l'utilisateur.**

### CP11 — Tirage pondéré par densité de routes (ablation)
- Poids = `valid_frac × (ε + densité)`, densité en km de route par km² et par dalle, précalculée par
  un script EdgeRunner dans un parquet à part (ne pas modifier les fichiers de road_graph_extractor).
- Option `data.density_eps: null` (paramètre de `IGNSamRoadDataset`) pour désactiver. **Fini quand** : les stats smoke montrent moins
  de patchs vides ; run d'ablation comparé à CP8.

### CP12 — Pont passant exactement sur un nœud
- Limite amont : `find_crossover_points` ignore les croisements en bout de segment. Avec
  `position_par_rapport_au_sol > 0`, ajouter explicitement les intersections pont × route comme
  croisements (paramètre supplémentaire de `WindowGraphLabeler`) et ne jamais fusionner les
  nœuds pont / route arrondis au même pixel.
- **Fini quand** : un test où le pont du toy passe par un carrefour n'a ni keypoint ni paire
  connectée à travers le croisement.

### CP13 — (Optionnel) Partir des checkpoints sam_road cityscale
- HF `congrui/sam_road`, entraînés à 1 m/px avec des patches de 512. Les offsets de TopoNet sont en
  pixels bruts : à 0.4 m ils sont 2.5 fois plus grands. Mettre les offsets à l'échelle ou
  ré-entraîner TopoNet. Priorité basse.

## 6. Pièges connus

- Le sous-module `sam` de sam_road est déclaré en SSH : forcer l'URL https après **chaque** clone
  (config git locale, non versionnée).
- `sam/segment_anything/predictor.py` importe `segment_anything` en absolu, donc les deux
  dossiers doivent être sur le `sys.path` (déjà fait dans `_sam_road.py`).
- `model.py` amont importe `wandb` au chargement : `sam_model()` injecte un module vide si wandb
  est absent (seule la viz amont, surchargée, l'utilise). Ne pas appeler le `validation_step` amont.
- affine ≥ 3 : utiliser `@` pour composer les transformations **et** pour les appliquer à des points.
- Workers en `forkserver` par défaut (`fork` est risqué une fois des threads pyarrow / GDAL
  lancés). Conséquences : `collate` au niveau du module, et le dataset se picke sans handles
  ouverts (`__getstate__`).
- RNG : `_check_process` réinitialise par processus (`torch.initial_seed()` + `RANK`) ;
  `sample_patch` amont est isolé par un contexte qui restaure le RNG global ; graine par index
  en éval, graines tirées du RNG du worker en train (séquence modifiée par la modularisation).
- Mémoire : sur l'A2000 locale (12 Go), seul 1024 px en batch 1 passe (avec
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). Pour le dev, utiliser 0.8 m / 512.
- Toy : le pont ne doit pas passer par les nœuds de la grille (cf. CP12).
- Avertissement Lightning « on_after_batch_transfer … LightningDataModule » : sans conséquence.
- DDP sous srun : chaque rang lance Hydra, donc fixer `hydra.run.dir=$OUT` (déjà fait dans le
  sbatch). Les clés absentes du YAML se surchargent avec `+clé=…` ; les autres avec `clé=…`.
- Paramètres de `IGNSamRoadDataset` absents du YAML (`min_valid`, `roads_cache_size`,
  `native_gsd`) : passer `+data.min_valid=0.3`, `make_loaders` les transmet au dataset.
- Ruff et pre-commit respectent le `.gitignore` : une règle non ancrée (`data/` au lieu de
  `/data/`) masquait tout `src/edgerunner/data/` au lint. Toujours ancrer les règles à la racine.
- torch est épinglé sur l'index PyTorch cu126 (`[tool.uv.sources]`). Si le driver des A40 est
  ≥ 580 (`nvidia-smi` sur un nœud), on peut passer en cu130 ; sinon ne pas retirer cet index.
- pyright ne comprend pas `rasterio.windows.Window(...)` (attrs ancienne syntaxe) : utiliser
  `Window.from_slices(rows, cols)` (voir `IGNSamRoadDataset._window`).
- `SAMRoadModule(..., provenance=...)` : la provenance est un hparam ; ne pas l'ajouter à
  `self.hparams` après coup, sinon `load_from_checkpoint` reçoit un argument inconnu.
- Proxy du cluster : ClearML a besoin de ses hôtes dans `no_proxy` ; NCCL a besoin de
  `NCCL_IB_DISABLE=1` et `NCCL_SOCKET_IFNAME=bond0` (déjà dans le sbatch).

## 7. Journal (ajouter en bas, ne pas réécrire)

| Date | CP | Qui | Résumé / mesures |
| --- | --- | --- | --- |
| 2026-10-28 | CP0–CP6 | Claude (Opus) | Pipeline écrit et testé sur le toy. Dataset : 49 ms/échantillon à 1024 px (1 processus), 110 échantillons/s avec 4 workers. A2000 : 1.65 it/s en batch 1 à 1024 (16-mixed), OOM en batch 2. Repère (autres labels, DeH4R) : 67 échantillons/s depuis le NVMe contre 4 depuis CIFS avec 8 workers. |
| 2026-10-28 | CP6b | Claude (Opus) | Simplification validée : 25 tests, `just dev`, mini-run (ckpt + TB), smoke toy 148 échantillons/s avec 4 workers. |
| 2026-10-28 | CP6c | Claude (Opus) | Bonnes pratiques : CI, pre-commit, pyright, provenance, cu126, licence MIT. 6 erreurs ruff masquées par l'ancien `.gitignore` corrigées. |
| 2026-10-28 | CP7 (local) | Claude (Opus) | Vraies dalles (index de test provisoire, 6 dalles) : smoke 109 ex/s, 3 époques en 15 min à 2 it/s, val_loss 0.092 → 0.071, road_iou@0.5 encore ≈ 0 (carte de route déjà visible sous 0.5) : entraînement trop court, à prolonger sur le cluster. |
| 2026-10-29 | Revue | Codex | Exclusion des pixels sans image des losses/IoU et des extrémités topo invalides ; cache raster LRU avec fermeture ; propagation des échecs rsync parallèles. 36 tests réussis, Ruff/formatage/Pyright OK, syntaxe Bash valide ; copie simulée hors SLURM. |
| 2026-10-29 | Modularisation | Codex | Losses indépendantes de SAM/Lightning, lecture/cache raster et sampling séparés, visibilité/collate typés, RNG amont isolé et script NVMe extrait. 52 tests CPU dont 12 unitaires sans dataset/checkpoint ; Ruff, formatage, Pyright et syntaxe Bash validés. SLURM non exécuté. |
| 2026-10-29 | CP6d | Claude (Opus) | Folds de road_graph_extractor reconstruits (259 dalles, 54 zones, 37 296 fenêtres, `eval_fold0` vérifié) dans `edgerunner.data.index` ; un `.slurm` par job sur 1 nœud × 3 A40 ; jobs simulés hors SLURM. Run local de nuit sur 190 dalles arrêté (il utilisait des folds provisoires incluant le fold `test`). |
| 2026-10-29 | CP6e | Claude (Opus) | Validation croisée 5 folds en job array (2 à la fois, 1 nœud × 3 A40 par fold), score par fold (`metrics.json`) et synthèse `cv_summary.csv` ; simulée sur le toy, jamais soumise au vrai SLURM. |
