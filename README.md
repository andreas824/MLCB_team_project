# MLCB Team Project — Communication-Aware ML for MDD

Re-analysis of the Maitra et al. (2023) snRNA-seq depression dataset
(GSE213982 + GSE144136, dlPFC, 71 donors, ~160k nuclei). We test whether
the female-dominant microglia signal (Mic1) and the male-dominant deep-layer
excitatory neuron signal (ExN10_L46) are two ends of a single
microglia↔neuron communication axis, visible only when cell-cell
communication is modelled explicitly.

## Pipeline overview

- **Phase A — Reproduction** *(this repo, in progress)*: load the combined
  count matrix, QC, normalize, Harmony integration, attach the authors'
  cell-type annotations, reproduce the Mic1 / ExN10_L46 DEG sanity checks,
  checkpoint to Drive.
- **Phase B — CellChat**: infer ligand-receptor networks per condition
  (MDD/Control × female/male); convert edge weights into per-donor
  communication features.
- **Phase C — ML**: pseudobulk per donor; XGBoost case/control classifier on
  expression + communication features; SHAP for an interpretable gene ranking.
- **Phase D — Evaluation**: donor-level stratified CV (no donor leakage);
  leave-one-sex-out as the direct test of the shared-axis hypothesis.

## Repository layout

```
MLCB_team_project/
├── src/
│   └── functions.py          # reusable pipeline functions (load_dataset, QC, ...)
├── notebooks/
│   └── reproduction.ipynb     # Phase A driver notebook (runs on Colab)
├── models/                    # saved models (gitignored — live on Drive)
├── data/                      # datasets (gitignored — live on Drive, NOT git)
├── .gitignore
└── README.md
```

**Code lives in git. Data does NOT.** The raw matrix is ~1.1 GB and is
re-downloadable from GEO, so it never belongs in the repository. It lives in
Google Drive and is fetched by the notebook.

## Where the data lives

Everything under `data/` and `models/` is on Google Drive, not GitHub:

```
/content/drive/MyDrive/MLCB_team_project/
├── data/
│   ├── raw/                   # GEO downloads (.mtx.gz, .csv.gz, SOFT)
│   └── checkpoints/           # AnnData .h5ad after each expensive step
└── models/                    # trained model .pkl files
```

To share data between team members, share this Drive folder (Share → Editor).
Alternatively, re-run the download cell — the data is public on GEO.

## Running Phase A (Colab)

1. **Runtime**: `Runtime > Change runtime type > CPU, High-RAM`.
   Phase A needs RAM (~160k nuclei), not GPU. Do **not** pick a GPU runtime.
2. Open `notebooks/reproduction.ipynb`.
3. Run the **boot cell** (mounts Drive, clones/pulls this repo, imports
   `functions`).
4. Run the pipeline cells top to bottom.
5. Checkpoints are written to Drive after each expensive step, so a Colab
   timeout never costs more than one step.

### The boot cell

```python
import os, sys
from google.colab import drive
drive.mount('/content/drive')

PROJECT_ROOT   = '/content/drive/MyDrive/MLCB_team_project'
RAW_DIR        = os.path.join(PROJECT_ROOT, 'data', 'raw')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'data', 'checkpoints')
for d in (RAW_DIR, CHECKPOINT_DIR):
    os.makedirs(d, exist_ok=True)

REPO_URL = 'https://github.com/andreas824/MLCB_team_project.git'
REPO_DIR = '/content/MLCB_team_project'
if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull

sys.path.insert(0, os.path.join(REPO_DIR, 'src'))
%load_ext autoreload
%autoreload 2
from functions import build_condition_map, load_dataset
```

## Editing code

The cycle is: edit `src/functions.py` in VSCode → `git commit` → `git push`
→ `git pull` in the Colab boot cell. `%autoreload 2` means no kernel restart
is needed after a pull. If Colab seems to run stale code, `Runtime > Restart`.

## Dataset facts (verified)

- Combined matrix already contains **both** sexes: 38 female + 33 male donors
  = **71 donors** (the matrix has 72 donor *tokens* because `M24_2` is a second
  run of donor `M24` — merged to one donor in `load_dataset`).
- Donor-level condition: female 20 MDD / 18 Control; male 17 MDD / 16 Control;
  total **37 MDD / 34 Control**.
- Cell labels are pre-annotated by the authors, encoded in the cell string
  `donor.barcode.broad.fine` (e.g. `F1.AAACCCACACCTCTGT-1.Mic.Mic1`), giving
  both the 7 broad classes and the 41 fine clusters for free.

## Reference

Maitra et al. (2023), *Nat. Commun.* 14, 2912.
GSE213982 (female) + GSE144136 (male).


