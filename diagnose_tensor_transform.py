#!/usr/bin/env python
"""
Find which tensor value-transform reproduces the frozen Phase B factors.

CHECK 3 showed every projected factor is INVERTED (negative r) vs the frozen
factors, and even the best matches are only |r| ~ 0.5-0.7. That pattern says
Giorgos used a different value column / transform than raw `magnitude_rank`.

This script rebuilds the tensor under several candidate transforms, runs the
same CP decomposition + NNLS projection for each, and reports how well each
reproduces the frozen factors (median |r| across factors, and Factor 4
specifically). The winner tells us what build_tensor should use.

Candidates tried:
    raw            : magnitude_rank as-is              (small = strong)  [current]
    one_minus      : 1 - magnitude_rank                (large = strong)
    neg_log10      : -log10(magnitude_rank + eps)      (large = strong, nonlinear)
    <rawcol>       : a raw magnitude column if present (lr_means / expr_prod / ...)

Usage
-----
    python diagnose_tensor_transform.py --checkpoint-dir data/checkpoints

If one candidate gives high POSITIVE median |r| (>~0.85) and a strong Factor 4,
set build_tensor's score handling to that transform and re-run preflight_check.
"""
from __future__ import annotations
import os, sys, argparse, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def build_tensor_with(liana_res, transform):
    """
    Same filtering/axes as run_rigorous_local.build_tensor, but applies a
    chosen transform to the value column before placing it in the tensor.

    transform : one of 'raw', 'one_minus', 'neg_log10', or a raw column name
                present in the table (e.g. 'lr_means', 'expr_prod').
    Returns (T, donor_order, value_col_used, transform_label).
    """
    import numpy as np
    import pandas as pd

    df = liana_res.copy()
    sample_col = 'donor_id' if 'donor_id' in df.columns else df.columns[0]
    if 'ligand_complex' in df.columns and 'receptor_complex' in df.columns:
        df['_lr'] = df['ligand_complex'].astype(str) + '^' + df['receptor_complex'].astype(str)
    elif 'ligand' in df.columns and 'receptor' in df.columns:
        df['_lr'] = df['ligand'].astype(str) + '^' + df['receptor'].astype(str)
    else:
        raise KeyError('cannot find ligand/receptor columns')
    send_col = 'source' if 'source' in df.columns else 'sender'
    recv_col = 'target' if 'target' in df.columns else 'receiver'

    # choose + transform the value column
    if transform in ('raw', 'one_minus', 'neg_log10'):
        base = 'magnitude_rank' if 'magnitude_rank' in df.columns else None
        if base is None:
            raise KeyError('magnitude_rank not in table')
        v = df[base].astype(float).values
        if transform == 'raw':
            vals = v
            label = 'magnitude_rank (raw)'
        elif transform == 'one_minus':
            vals = 1.0 - v
            label = '1 - magnitude_rank'
        else:
            vals = -np.log10(v + 1e-12)
            label = '-log10(magnitude_rank)'
        value_col = base
    else:
        # treat `transform` as a raw column name
        if transform not in df.columns:
            raise KeyError(f'{transform} not in table')
        vals = df[transform].astype(float).values
        value_col = transform
        label = f'{transform} (raw column)'
    df['_val'] = vals

    # outer_cells filter: pairs present in ALL donors
    n_donors_total = df[sample_col].nunique()
    per_lr = df.groupby('_lr')[sample_col].nunique()
    keep = set(per_lr[per_lr == n_donors_total].index)
    df = df[df['_lr'].isin(keep)].copy()

    donors = sorted(df[sample_col].astype(str).unique())
    lrs = sorted(df['_lr'].unique())
    sends = sorted(df[send_col].astype(str).unique())
    recvs = sorted(df[recv_col].astype(str).unique())
    di = {d: i for i, d in enumerate(donors)}
    li_ = {x: i for i, x in enumerate(lrs)}
    si = {x: i for i, x in enumerate(sends)}
    ri = {x: i for i, x in enumerate(recvs)}

    T = np.full((len(donors), len(lrs), len(sends), len(recvs)), np.nan, float)
    T[df[sample_col].astype(str).map(di).values,
      df['_lr'].map(li_).values,
      df[send_col].astype(str).map(si).values,
      df[recv_col].astype(str).map(ri).values] = df['_val'].values
    return T, donors, value_col, label


def score_transform(T, donor_order, FB, rank, n_iter_max):
    """Run CP + NNLS projection, return (median_abs_r, factor4_abs_r, detail)."""
    import numpy as np
    import pandas as pd
    from src.tensor_features import TensorFactorProjector, _cp_cache_clear
    _cp_cache_clear()

    proj = TensorFactorProjector(tensor=T, donor_order=donor_order, rank=rank,
                                 random_state=0, n_iter_max=n_iter_max)
    X_all = pd.DataFrame({'_d': np.arange(len(donor_order))}, index=donor_order)
    proj.fit(X_all)
    Fp = proj.transform(X_all)

    common = [d for d in Fp.index if d in FB.index]
    P = Fp.loc[common].values
    B = FB.loc[common].values
    rk = min(P.shape[1], B.shape[1])

    corr = np.zeros((rk, rk))
    for i in range(rk):
        for j in range(rk):
            c = np.corrcoef(B[:, i], P[:, j])[0, 1]
            corr[i, j] = c if np.isfinite(c) else 0.0
    used, best = set(), []
    for i in range(rk):
        order = np.argsort(-np.abs(corr[i]))
        j = next((jj for jj in order if jj not in used), order[0])
        used.add(j)
        best.append((i, j, corr[i, j]))
    med = float(np.median([abs(c) for _, _, c in best]))
    f4 = next((abs(c) for i, j, c in best if i == 3), float('nan'))
    return med, f4, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir', default='data/checkpoints')
    ap.add_argument('--rank', type=int, default=5)
    ap.add_argument('--n-iter-max', type=int, default=50)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(os.path.join(here, 'src')) and here not in sys.path:
        sys.path.insert(0, here)

    cdir = args.checkpoint_dir
    liana = pd.read_parquet(os.path.join(cdir, 'phaseB_liana_per_donor.parquet'))
    FB = pd.read_parquet(os.path.join(cdir, 'phaseB_donor_factors.parquet'))
    FB.index = FB.index.astype(str)

    print('columns in liana table:')
    print('  ', list(liana.columns))
    # candidate transforms: the three magnitude_rank variants, plus any raw
    # magnitude-like column that exists in the table
    candidates = ['raw', 'one_minus', 'neg_log10']
    for col in ('lr_means', 'expr_prod', 'lrscore', 'lr_logfc', 'magnitude'):
        if col in liana.columns:
            candidates.append(col)

    print(f'\nTrying {len(candidates)} transforms '
          f'(CP rank={args.rank}, n_iter_max={args.n_iter_max}). '
          f'~{args.n_iter_max}s each...\n')
    print(f'{"transform":28s} {"median|r|":>10s} {"Factor4|r|":>11s}   verdict')
    print('-' * 70)

    results = []
    for t in candidates:
        try:
            T, donor_order, vcol, label = build_tensor_with(liana, t)
        except KeyError as e:
            print(f'{t:28s} {"--":>10s} {"--":>11s}   skip ({e})')
            continue
        med, f4, best = score_transform(T, donor_order, FB,
                                        args.rank, args.n_iter_max)
        verdict = ('MATCH' if med > 0.85 else
                   'partial' if med > 0.7 else 'no')
        flag4 = '*' if (f4 == f4 and f4 > 0.8) else ''
        print(f'{label:28s} {med:>10.3f} {f4:>11.3f}{flag4:1s}  {verdict}')
        results.append((label, med, f4, best))

    if results:
        best_overall = max(results, key=lambda x: x[1])
        print('\n' + '=' * 70)
        print(f'BEST TRANSFORM: "{best_overall[0]}"  '
              f'(median |r| = {best_overall[1]:.3f}, '
              f'Factor 4 |r| = {best_overall[2]:.3f})')
        print('=' * 70)
        print('\nPer-factor detail for the best transform:')
        for i, j, c in best_overall[3]:
            tag = 'OK' if abs(c) > 0.7 else ('weak' if abs(c) > 0.4 else 'BAD')
            print(f'  frozen F{i+1} <-> proj F{j+1}   r = {c:+.3f}  [{tag}]')

        if best_overall[1] > 0.85:
            print(f'\n=> Update build_tensor to use this transform, then '
                  f're-run preflight_check.py to confirm.')
        else:
            print(f'\n=> Even the best transform is imperfect (median |r| '
                  f'{best_overall[1]:.3f}). Likely Giorgos did something this '
                  f'script does not cover (different filter, fillna, or '
                  f'normalization). Send him the NOTE and ask for the exact '
                  f'tensor-build line. Higher n_iter_max may also help if it '
                  f'is a convergence issue (try --n-iter-max 200).')


if __name__ == '__main__':
    main()