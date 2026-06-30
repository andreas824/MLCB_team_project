# MLCB Team Project — Communication-Aware ML for MDD

Re-analysis of the Maitra et al. (2023) snRNA-seq depression dataset
(GSE213982 + GSE144136, dlPFC, 71 donors, ~160k nuclei). We test whether
the female-dominant microglia signal (`Mic1`) and the male-dominant deep-layer
excitatory-neuron signal (`ExN10_L46`) are two ends of a single
**microglia → neuron communication axis** that is visible only when cell–cell
communication is modelled explicitly.

The full write-up — methods, results, figures and honest conclusion — is in
[`report/MLCB_report.pdf`](report/MLCB_report.pdf) (source:
[`report/MLCB_report.tex`](report/MLCB_report.tex)). Authors: Giorgos
Boulogeorgos, Andreas Mici.

**Headline result.** A single data-driven communication program (*Factor 4*, an
endothelial+microglia → `ExN10_L46` axis) is the dominant, cohort-clean,
MDD-leaning communication signal and is elevated in MDD donors of *both* sexes —
**directionally supporting** the shared-axis hypothesis. But when the factors
compete head-to-head with microglial/neuronal gene expression, feature selection
keeps only genes and drops every factor: at *n* = 71 the communication
representation adds **no predictive value independent of expression**. The
biology is real; it is a low-rank shadow of microglial transcriptional state, not
an independent signal.

## Pipeline overview

The analysis runs in three phases plus an interpretability layer (see Figure 1 of
the report):

- **Phase A — Reproduction** (`notebooks/reproduction.ipynb`, `src/functions.py`):
  stream the ~1.1 GB combined matrix into `AnnData`, QC, normalize to CP10⁴ +
  `log1p` (raw counts kept in `.layers['counts']` for pseudobulk), merge the two
  `M24` runs into one donor, attach sex/diagnosis and the authors' baked-in
  cell-type labels, and run a `Mic1` / `ExN10_L46` DEG sanity check. Checkpoints
  to Drive.
- **Phase B — Cell–cell communication** (`notebooks/cellchat.ipynb`,
  `src/phaseB_step3_local.py`): per-donor ligand–receptor inference with
  **LIANA** (`rank_aggregate.by_sample`) on an 8-node hybrid label set
  (`Ast, End, ExN10_L46, ExN_other, InN, Mic, OPC, Oli`); arrange the scores as a
  4-D tensor (71 donors × 1650 L–R pairs × 8 senders × 8 receivers, masked
  `NaN`s, `−log₁₀(magnitude_rank + ε)` transform); compress with a **non-negative
  CP/PARAFAC decomposition (rank 5)** into 5 communication *programs*. The
  hypothesis program is *Factor 4*.
  > Note: `cellchat.ipynb` keeps its name from the original plan but actually
  > implements LIANA — the project deliberately moved off CellChat/R.
- **Phase C — Machine learning** (`run_rigorous_local.py`, `src/rncv.py`,
  `src/pseudobulk.py`, `src/feature_selection.py`, `src/estimators.py`): build
  per-cell-type pseudobulk and compare **three feature sets** — expression-only,
  communication-only (the 5 factors), and combined — under **repeated nested
  cross-validation** (3 rounds × 5 outer × 3 inner, 20 Optuna trials, 5
  estimators: LR/GNB/LDA/RF/XGB), with `VarianceTopK(1000)` → mRMR (`k=15`) fit
  in-fold and bootstrap 95% CIs. The tensor is **re-fit inside every outer-train
  fold** (`src/tensor_features.py`, `TensorFactorProjector`) so the communication
  features are leakage-safe.
- **Step 4 — Interpretability & hypothesis test** (`src/shap_step4.py`): SHAP on
  the models (standalone / combined / in-fold leakage-safe) plus the directed,
  per-sex test of whether *Factor 4* (microglia → `ExN10_L46`) matters in both
  sexes.

> **Evaluation design (important).** Sex is perfectly confounded with cohort
> (females = GSE213982, males = GSE144136), so we do **not** use leave-one-sex-out
> (it would be a train/test-across-batch split). Instead we use **within-sex
> pooled donor-level CV stratified by sex × diagnosis**, which tests whether
> communication *adds value* and whether *Factor 4* is shared across sexes.

## Repository layout

```
MLCB_team_project/
├── src/
│   ├── functions.py            # Phase A/B helpers (load_dataset, cell-string parse, QC, …)
│   ├── phaseB_step3_local.py   # Phase B local runner: LIANA tensor + CP decomposition (+ elbow)
│   ├── tensor_features.py      # TensorFactorProjector — in-fold (leakage-safe) CP refit
│   ├── pseudobulk.py           # per-cell-type pseudobulk expression features
│   ├── feature_selection.py    # VarianceTopK + MRMRSelector (fit in-fold)
│   ├── estimators.py           # LR / GNB / LDA / RF / XGB factories
│   ├── rncv.py                 # repeated nested CV driver
│   ├── metrics.py              # AUC / MCC + bootstrap CIs
│   ├── preprocessing.py        # scaling / transforms
│   └── shap_step4.py           # Step 4: SHAP + directed per-sex hypothesis test
├── notebooks/
│   ├── reproduction.ipynb      # Phase A driver (Colab)
│   ├── cellchat.ipynb          # Phase B driver — LIANA per-donor (despite the name)
│   └── ml.ipynb                # Phase C / Step 4 exploration
├── run_rigorous_local.py       # Phase C: rigorous repeated-nested-CV runner (in-fold tensor refit)
├── preflight_check.py          # ~1-min sanity check before the full Phase C run
├── diagnose_tensor_transform.py# diagnostic for the magnitude_rank → −log10 transform
├── requirements-local.txt      # local (off-Colab) Phase B/C environment
├── report/                     # MLCB_report.tex/.pdf + figures (the deliverable)
├── data/                       # datasets + checkpoints (gitignored — live on Drive)
├── .gitignore
└── README.md
```

**Code lives in git. Data does NOT.** The raw matrix is ~1.1 GB and is
re-downloadable from GEO, so it never belongs in the repository. It lives on
Google Drive and is fetched by the notebook.

## Where the data lives

Everything under `data/` (and any `models/`) is on Google Drive, not GitHub:

```
/content/drive/MyDrive/MLCB_team_project/
└── data/
    ├── raw/                    # GEO downloads (.mtx.gz, .csv.gz, SOFT)
    └── checkpoints/            # AnnData .h5ad + small .parquet/.json artefacts
```

The small checkpoints that the local Phase B/C runners consume are:

| File | Produced by | Used by |
|---|---|---|
| `phaseA_final_for_cellchat.h5ad` | Phase A notebook | Phase B (LIANA) |
| `phaseA_obs.parquet` | Phase A | donor metadata / labels |
| `phaseB_liana_per_donor.parquet` | Phase B (LIANA per donor) | tensor construction |
| `phaseB_donor_factors.parquet` | Phase B (CP decomposition) | exploratory factor loadings |
| `phaseC_pseudobulk_percelltype.parquet` | Phase C pseudobulk | expression / combined sets |

To share data between team members, share the Drive folder (Share → Editor), or
re-run the download cell — the data is public on GEO.

## Reproducing the analysis

### Phase A — reproduction (Colab)

1. **Runtime**: `Runtime > Change runtime type > CPU, High-RAM`. Phase A needs RAM
   (~160k nuclei), not GPU. Do **not** pick a GPU runtime.
2. Open `notebooks/reproduction.ipynb`.
3. Run the **boot cell** (below): it mounts Drive, clones/pulls this repo, and
   imports `functions`.
4. Run the pipeline cells top to bottom. Checkpoints are written to Drive after
   each expensive step, so a Colab timeout never costs more than one step.

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

### Phase B + C + Step 4 — local (off-Colab)

Phase B Step 3 (LIANA tensor / CP decomposition) and the rigorous Phase C CV are
run locally (built for a multi-core laptop, e.g. Apple Silicon). They need
**only the small checkpoints**, not the multi-GB `.h5ad`.

```bash
conda create -n mlcb python=3.11 -y && conda activate mlcb
pip install -r requirements-local.txt
# Apple Silicon, if XGBoost complains about OpenMP:  brew install libomp

# Phase B: per-donor LIANA tensor + CP decomposition (with reconstruction-error elbow)
python src/phaseB_step3_local.py --checkpoint-dir data/checkpoints --elbow

# Phase C: sanity-check (~1 min), then the rigorous run (~50 min)
python preflight_check.py --checkpoint-dir data/checkpoints
python run_rigorous_local.py --sets comm expr combined

# Step 4: SHAP + directed per-sex hypothesis test
python src/shap_step4.py
```

Headline settings: **seed 42**, 3 × 5 × 3 repeated nested CV, 20 Optuna trials,
per-fold tensor refit; the CP decomposition itself uses `random_state=0`. Phase B
was produced with LIANA 1.7.3 / cell2cell 0.8.4. The tensor reconstruction-error
curve (`report/figures/phaseB_reconstruction_error.png`) shows the error declines
*smoothly* with no sharp elbow; rank 5 is a parsimony/interpretability choice,
validated post-hoc by clean batch/biology separation and in-fold reproducibility
(see the report's "Why rank 5").

## Editing code

Edit `src/*.py` in your editor → `git commit` → `git push` → `git pull` in the
Colab boot cell. `%autoreload 2` means no kernel restart after a pull. If Colab
seems to run stale code, `Runtime > Restart`.

## Dataset facts (verified)

- The combined matrix already contains **both** sexes: 38 female + 33 male donors
  = **71 donors** (the matrix has 72 donor *tokens* because `M24_2` is a second
  run of donor `M24` — merged to one donor in `load_dataset`).
- Donor-level condition: female 20 MDD / 18 Control; male 17 MDD / 16 Control;
  total **37 MDD / 34 Control**. 160,711 high-quality nuclei, 36,588 genes
  (156,911 nuclei after dropping `Mix` for the communication graph).
- Cell labels are pre-annotated by the authors, encoded in the cell string
  `donor.barcode.broad.fine` (e.g. `F1.AAACCCACACCTCTGT-1.Mic.Mic1`), giving both
  the broad classes and the fine clusters for free.
- **Confound to track:** sex is perfectly confounded with cohort/library
  (females = GSE213982, males = GSE144136) — the reason evaluation is within-sex,
  not leave-one-sex-out.

## Reference

Maitra et al. (2023), *Nat. Commun.* **14**, 2912.
GSE213982 (female) + GSE144136 (male).
