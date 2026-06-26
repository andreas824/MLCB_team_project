#!/usr/bin/env python
"""
Rigorous donor-level CV for the MDD communication-aware classification,
run locally (built for a multi-core laptop, not Colab).

Communication features are recomputed inside every CV fold: the tensor is
refit on the training donors only and test donors are projected onto the
fixed factors (TensorFactorProjector). This removes the mild leak of using
the all-donor phaseB_donor_factors.parquet. Expression is per-celltype
pseudobulk with a variance pre-filter fit in-fold. Stratification is by
sex x diagnosis. All five estimators are tuned via the nested Optuna loop,
with bootstrap CIs.

Only the small checkpoints are needed (NOT the 5.8 GB h5ad), because the
per-celltype pseudobulk is read from its cache:
    - phaseB_liana_per_donor.parquet   (per-donor LIANA, for the tensor)
    - phaseA_obs.parquet               (donor metadata / labels)
    - phaseC_pseudobulk_percelltype.parquet  (cached expression; optional,
      only needed for the expression / combined sets)

Usage
-----
    python run_rigorous_local.py --sets comm
    python run_rigorous_local.py --sets comm expr combined
    python run_rigorous_local.py --sets comm --n-rounds 2 --n-jobs 6

Environment (Python >= 3.10)
----------------------------
    conda create -n mlcb python=3.11 -y && conda activate mlcb
    pip install scanpy pyarrow xgboost shap scikit-learn optuna \
                mrmr_selection tensorly
    # cell2cell is NOT required: the projector uses tensorly directly.
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '1')      # avoid BLAS oversubscription;
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # parallelism comes from n_jobs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint-dir', default='data/checkpoints',
                    help='folder with the phaseB_/phaseA_ parquet checkpoints')
    ap.add_argument('--sets', nargs='+', default=['comm'],
                    choices=['comm', 'expr', 'combined'],
                    help='which feature sets to run')
    ap.add_argument('--rank', type=int, default=5)
    ap.add_argument('--n-rounds', type=int, default=3)
    ap.add_argument('--n-outer', type=int, default=5)
    ap.add_argument('--n-inner', type=int, default=3)
    ap.add_argument('--n-trials', type=int, default=20)
    ap.add_argument('--n-var-prefilter', type=int, default=1000)
    ap.add_argument('--k-features', type=int, default=15)
    ap.add_argument('--n-jobs', type=int, default=4,
                    help='parallel jobs for estimators (use ~half your threads)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-iter-max', type=int, default=50,
                    help='max iterations for the per-fold CP decomposition '
                         '(50 is plenty; CP converges well before then)')
    args = ap.parse_args()

    # local imports (after warnings/env are set)
    import numpy as np
    import pandas as pd

    # make src/ importable whether run from repo root or elsewhere
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (here, os.path.join(here, 'src'), os.path.join(here, '..')):
        if os.path.isdir(os.path.join(cand, 'src')) and cand not in sys.path:
            sys.path.insert(0, cand)
    if os.path.isdir(os.path.join(here, 'src')) and here not in sys.path:
        sys.path.insert(0, here)

    from src.rncv import RepeatedNestedCV
    from src.estimators import get_estimators_and_spaces
    from src.feature_selection import MRMRSelector, VarianceTopK
    from src.tensor_features import TensorFactorProjector

    cdir = args.checkpoint_dir
    RESULTS_DIR = os.path.join(cdir, 'phaseC_results_rigorous')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- metadata + labels + stratification --------------------------------
    obs = pd.read_parquet(os.path.join(cdir, 'phaseA_obs.parquet'))
    meta = obs.drop_duplicates('donor_id').set_index('donor_id')[['condition', 'sex']]
    meta.index = meta.index.astype(str)
    meta['y'] = (meta['condition'] == 'MDD').astype(int)
    meta['strat'] = meta['sex'].astype(str) + '_' + meta['condition'].astype(str)
    print('sex x diagnosis:', meta['strat'].value_counts().to_dict())

    # ---- build the 4D tensor ONCE from per-donor LIANA ---------------------
    # tensorly path: we reconstruct the dense tensor + donor order without
    # cell2cell. liana's to_tensor_c2c would also work but pulls cell2cell;
    # here we pivot the long table directly.
    print('\nbuilding 4D tensor from per-donor LIANA ...', flush=True)
    liana_res = pd.read_parquet(os.path.join(cdir, 'phaseB_liana_per_donor.parquet'))
    T, donor_order = build_tensor(liana_res)
    print(f'tensor: {T.shape} | donors: {len(donor_order)} '
          f'| masked cells: {int(np.isnan(T).sum()):,}')

    # ---- expression (only if needed) ---------------------------------------
    expr = None
    if 'expr' in args.sets or 'combined' in args.sets:
        pb_cache = os.path.join(cdir, 'phaseC_pseudobulk_percelltype.parquet')
        if not os.path.exists(pb_cache):
            sys.exit(f'ERROR: {pb_cache} not found. Download the cached '
                     f'pseudobulk from Drive, or run the pseudobulk step first.')
        expr = pd.read_parquet(pb_cache)
        expr.index = expr.index.astype(str)
        print(f'expression (cached pseudobulk): {expr.shape}')

    # ---- common donors + X (donor-identity frame) --------------------------
    donors = [d for d in donor_order if d in meta.index]
    if expr is not None:
        donors = [d for d in donors if d in expr.index]
    print(f'common donors: {len(donors)}')
    y = meta.loc[donors, 'y'].values
    strat = meta.loc[donors, 'strat'].values
    # X carries only donor identity; the projector recomputes factors per fold
    X_id = pd.DataFrame({'_donor': np.arange(len(donors))}, index=donors)

    ests, spaces = get_estimators_and_spaces()

    def run(name, prefilter, selector, X):
        print(f'\n========== {name}  ({args.n_rounds}r x {args.n_outer}o x '
              f'{args.n_inner}i x {args.n_trials}t, n_jobs={args.n_jobs}) ==========',
              flush=True)
        t0 = time.time()
        rncv = RepeatedNestedCV(
            estimators=ests, param_spaces=spaces,
            prefilter=prefilter, feature_selector=selector,
            tune_hyperparams=True,
            n_rounds=args.n_rounds, n_outer=args.n_outer,
            n_inner=args.n_inner, n_trials=args.n_trials,
            random_state=args.seed, n_jobs=args.n_jobs, verbose=False,
        )
        rncv.fit_evaluate(X, y, stratify_labels=strat)
        summ = rncv.summarize()
        tag = name.lower().replace('-', '_').replace(' ', '')
        perfold = pd.concat({a: rncv.results_[a] for a in rncv.results_},
                            names=['algorithm'])
        perfold.to_parquet(os.path.join(RESULTS_DIR, f'perfold_{tag}.parquet'))
        summ.to_parquet(os.path.join(RESULTS_DIR, f'summary_{tag}.parquet'))
        dt = time.time() - t0
        print(f'  [{dt:.0f}s = {dt/60:.1f} min]  AUC median [95% CI]:')
        for a in rncv.results_:
            m, lo, hi = (summ.loc[a, ('AUC', c)] for c in ('median', 'ci_low', 'ci_high'))
            mc = summ.loc[a, ('MCC', 'median')]
            print(f'    {a:4s}  AUC={m:.3f} [{lo:.3f}, {hi:.3f}]   MCC={mc:.3f}')
        return rncv

    results = {}

    if 'comm' in args.sets:
        proj = TensorFactorProjector(tensor=T, donor_order=donor_order,
                                     rank=args.rank, random_state=0,
                                     n_iter_max=args.n_iter_max)
        # projector produces the 5 factors; no further selection needed
        results['comm'] = run('communication', prefilter=proj,
                              selector=None, X=X_id)

    if 'expr' in args.sets:
        # expression genes by donor; static pseudobulk is leakage-safe to hold,
        # variance prefilter + mRMR are fit in-fold.
        Xe = expr.loc[donors]
        results['expr'] = run('expression',
                              prefilter=VarianceTopK(n_top=args.n_var_prefilter),
                              selector=MRMRSelector(k=args.k_features, relevance='mi'),
                              X=Xe)

    if 'combined' in args.sets:
        proj = TensorFactorProjector(tensor=T, donor_order=donor_order,
                                     rank=args.rank, random_state=0,
                                     n_iter_max=args.n_iter_max)
        assembler = CombinedFactorsGenes(
            projector=proj, genes_df=expr.loc[donors],
            n_var_prefilter=args.n_var_prefilter)
        # assembler yields [Factor_1..5, prefiltered genes]; mRMR selects from all,
        # factors protected so they always reach the selector.
        results['combined'] = run('combined', prefilter=assembler,
                                  selector=MRMRSelector(k=args.k_features, relevance='mi'),
                                  X=X_id)

    # ---- value-add summary if both expr and combined ran -------------------
    if 'expr' in results and 'combined' in results:
        print('\n\n=========== VALUE-ADD: COMBINED vs EXPRESSION ===========')
        se, sc_ = results['expr'].summarize(), results['combined'].summarize()
        print(f'{"algo":6s} {"expr AUC":18s} {"combined AUC":18s} dAUC   sep?')
        for a in se.index:
            em, el, eh = (se.loc[a, ('AUC', c)] for c in ('median', 'ci_low', 'ci_high'))
            cm, cl, ch = (sc_.loc[a, ('AUC', c)] for c in ('median', 'ci_low', 'ci_high'))
            sep = 'YES' if cl > eh else ('~' if cm > em else 'no')
            print(f'{a:6s} {em:.3f}[{el:.3f},{eh:.3f}]  {cm:.3f}[{cl:.3f},{ch:.3f}]  '
                  f'{cm-em:+.3f}  {sep}')

    print(f'\nResults saved to: {RESULTS_DIR}')


# =====================================================================
# Tensor construction from the long per-donor LIANA table (no cell2cell)
# =====================================================================

def build_tensor(liana_res):
    """
    Build a dense 4D tensor (donor x LR x sender x receiver) from the long
    per-donor LIANA result, with NaN where a (donor, sender, receiver) block
    was masked (cell type below min_cells in that donor).

    Returns (T, donor_order). Mirrors li.multi.to_tensor_c2c(how='outer_cells')
    closely enough for the factorization: same axes, magnitude_rank as values,
    union of all coordinates, missing combos left NaN.
    """
    import numpy as np
    import pandas as pd

    df = liana_res.copy()
    # column names in liana output
    sample_col = 'donor_id' if 'donor_id' in df.columns else df.columns[0]
    # ligand-receptor identity
    if 'ligand_complex' in df.columns and 'receptor_complex' in df.columns:
        df['_lr'] = df['ligand_complex'].astype(str) + '^' + df['receptor_complex'].astype(str)
    elif 'ligand' in df.columns and 'receptor' in df.columns:
        df['_lr'] = df['ligand'].astype(str) + '^' + df['receptor'].astype(str)
    else:
        raise KeyError('cannot find ligand/receptor columns in liana_res')
    send_col = 'source' if 'source' in df.columns else 'sender'
    recv_col = 'target' if 'target' in df.columns else 'receiver'
    score_col = 'magnitude_rank' if 'magnitude_rank' in df.columns else 'lr_means'

    # --- reproduce liana's to_tensor_c2c(how='outer_cells') filtering -------
    # 'outer_cells' keeps only LR pairs present across ALL donors (samples).
    # The per-donor LIANA table is dense within each donor's detected pairs;
    # restricting to pairs seen in every donor reduces 4471 -> the Phase B
    # 1650 pairs, matching the tensor the frozen factors were fit on.
    n_donors_total = df[sample_col].nunique()
    per_lr_donors = df.groupby('_lr')[sample_col].nunique()
    keep_lrs = set(per_lr_donors[per_lr_donors == n_donors_total].index)
    n_before = df['_lr'].nunique()
    df = df[df['_lr'].isin(keep_lrs)].copy()
    print(f'  LR pairs: {n_before} raw -> {len(keep_lrs)} present in all '
          f'{n_donors_total} donors (outer_cells filter)')

    donors = sorted(df[sample_col].astype(str).unique())
    lrs    = sorted(df['_lr'].unique())
    sends  = sorted(df[send_col].astype(str).unique())
    recvs  = sorted(df[recv_col].astype(str).unique())

    di = {d: i for i, d in enumerate(donors)}
    li_ = {x: i for i, x in enumerate(lrs)}
    si = {x: i for i, x in enumerate(sends)}
    ri = {x: i for i, x in enumerate(recvs)}

    T = np.full((len(donors), len(lrs), len(sends), len(recvs)),
                np.nan, dtype=float)
    d_idx = df[sample_col].astype(str).map(di).values
    l_idx = df['_lr'].map(li_).values
    s_idx = df[send_col].astype(str).map(si).values
    r_idx = df[recv_col].astype(str).map(ri).values
    vals  = df[score_col].astype(float).values
    T[d_idx, l_idx, s_idx, r_idx] = vals
    return T, donors


# =====================================================================
# Combined assembler: in-fold projected factors + prefiltered genes
# =====================================================================

class _CombinedBase:
    pass


def _make_combined_class():
    from sklearn.base import BaseEstimator, TransformerMixin
    import numpy as np
    import pandas as pd

    class CombinedFactorsGenes(BaseEstimator, TransformerMixin):
        """
        Concatenate in-fold projected communication factors with a variance-
        prefiltered slice of the per-donor gene expression.

        X index = donor_id. genes_df is the full per-donor pseudobulk matrix
        (static; per-donor pseudobulk is leakage-safe to hold). The projector
        recomputes the factors per fold; genes are looked up by donor_id and
        variance-filtered on the training fold only.
        """
        def __init__(self, projector, genes_df, n_var_prefilter=1000):
            self.projector = projector
            self.genes_df = genes_df
            self.n_var_prefilter = n_var_prefilter

        def fit(self, X, y=None):
            self.projector.fit(X, y)
            ids = [str(d) for d in X.index]
            g_tr = self.genes_df.loc[ids]
            v = g_tr.var(axis=0).values
            n_keep = min(self.n_var_prefilter, g_tr.shape[1])
            keep = np.sort(np.argsort(v)[::-1][:n_keep])
            self.gene_cols_ = [g_tr.columns[i] for i in keep]
            return self

        def transform(self, X):
            fac = self.projector.transform(X)            # (n, rank) DataFrame
            ids = [str(d) for d in X.index]
            g = self.genes_df.loc[ids, self.gene_cols_].copy()
            g.index = X.index
            return pd.concat([fac, g], axis=1)

        def get_support(self):
            return list(self.projector.feature_names_out_) + list(self.gene_cols_)

    return CombinedFactorsGenes


CombinedFactorsGenes = _make_combined_class()


if __name__ == '__main__':
    main()