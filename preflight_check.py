#!/usr/bin/env python
"""
Pre-flight check for the rigorous communication run. Runs in ~1 minute and
validates three things BEFORE the ~50-minute full run:

  1. Tensor builds correctly: shape (71, 1650, 8, 8), outer_cells filter works.
  2. One CP refit on ALL donors runs and produces sane factors.
  3. CRITICAL: the in-fold projected factors agree with the frozen Phase B
     factors (phaseB_donor_factors.parquet). If the per-factor correlations
     are high (|r| > ~0.7), the tensor values are on the same footing as
     Phase B and the rigorous run is comparable to the pragmatic one. If they
     are near zero or systematically negative, the tensor `score_col`
     (magnitude_rank) is mis-scaled / inverted relative to what Giorgos used
     in Phase B -- which would explain LR/LDA collapsing while GNB/XGB hold.

Usage
-----
    python preflight_check.py --checkpoint-dir data/checkpoints

If check 3 fails, do NOT run the full rigorous job yet: first confirm with
Giorgos which column Phase B used as tensor values (magnitude_rank vs
1-magnitude_rank vs lr_means / expr_prod) and align build_tensor's score_col
to match it.
"""
from __future__ import annotations
import os, sys, argparse, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir', default='data/checkpoints')
    ap.add_argument('--rank', type=int, default=5)
    ap.add_argument('--n-iter-max', type=int, default=50)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    # make src/ importable
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(os.path.join(here, 'src')) and here not in sys.path:
        sys.path.insert(0, here)

    # reuse the EXACT build_tensor + projector that the real run uses
    from run_rigorous_local import build_tensor
    from src.tensor_features import TensorFactorProjector, _cp_cache_clear
    _cp_cache_clear()

    cdir = args.checkpoint_dir

    # ---- 1. build tensor ---------------------------------------------------
    print('=' * 64)
    print('CHECK 1 — tensor construction')
    print('=' * 64)
    liana = pd.read_parquet(os.path.join(cdir, 'phaseB_liana_per_donor.parquet'))
    T, donor_order = build_tensor(liana)
    print(f'tensor shape : {T.shape}')
    print(f'donors       : {len(donor_order)}')
    print(f'masked cells : {int(np.isnan(T).sum()):,} '
          f'({100*np.isnan(T).mean():.1f}%)')
    ok_shape = (T.ndim == 4 and T.shape[0] == len(donor_order))
    ok_pairs = (T.shape[1] == 1650)
    print(f'  [{"PASS" if ok_shape else "FAIL"}] 4D, donor axis matches order')
    print(f'  [{"PASS" if ok_pairs else "WARN"}] 1650 LR pairs '
          f'(got {T.shape[1]}; Phase B expected 1650)')
    if not ok_pairs:
        print('       -> if not 1650, the outer_cells filter or the LIANA '
              'table differs from Phase B. Investigate before proceeding.')

    # value-column sanity: magnitude_rank lives in (0,1], small = strong
    print(f'\ntensor value range (observed): '
          f'[{np.nanmin(T):.4g}, {np.nanmax(T):.4g}], '
          f'median {np.nanmedian(T):.4g}')
    print('  note: liana magnitude_rank is a RANK in (0,1], small = strong '
          'signal. If Phase B used the rank directly, factors load on WEAK '
          'communication. Check 3 tells us if this matters.')

    # ---- 2. one CP refit on all donors ------------------------------------
    print('\n' + '=' * 64)
    print('CHECK 2 — single CP decomposition (all donors)')
    print('=' * 64)
    import time
    proj = TensorFactorProjector(tensor=T, donor_order=donor_order,
                                 rank=args.rank, random_state=0,
                                 n_iter_max=args.n_iter_max)
    X_all = pd.DataFrame({'_d': np.arange(len(donor_order))}, index=donor_order)
    t0 = time.time()
    proj.fit(X_all)
    dt = time.time() - t0
    A = proj.factors_[0]    # donor loadings from the decomposition itself
    print(f'fit time          : {dt:.1f}s  (x60 refits => ~{dt*60/60:.0f} min '
          f'for the full run)')
    print(f'donor-factor A    : {A.shape}')
    print(f'all factors >= 0  : {all((np.asarray(f) >= -1e-9).all() for f in proj.factors_)}')

    # projected factors via NNLS (what the test fold actually receives)
    F_proj = proj.transform(X_all)     # (71, rank) DataFrame
    print(f'projected factors : {F_proj.shape}')

    # ---- 3. compare to frozen Phase B factors -----------------------------
    print('\n' + '=' * 64)
    print('CHECK 3 — agreement with frozen Phase B factors  [CRITICAL]')
    print('=' * 64)
    fb_path = os.path.join(cdir, 'phaseB_donor_factors.parquet')
    if not os.path.exists(fb_path):
        print(f'SKIP: {fb_path} not found. Cannot compare. Download the frozen '
              f'factors from Drive to run this check.')
        return

    FB = pd.read_parquet(fb_path)
    FB.index = FB.index.astype(str)
    # align donors
    common = [d for d in F_proj.index if d in FB.index]
    print(f'common donors     : {len(common)} / {len(F_proj)}')
    Fp = F_proj.loc[common].values
    Fb = FB.loc[common].values
    rk = min(Fp.shape[1], Fb.shape[1])

    # CP factors are permutation- + scale-ambiguous, so match each frozen
    # factor to its best-correlated projected factor (Hungarian-free greedy).
    print(f'\nMatching {rk} frozen factors to projected factors '
          f'(|correlation|, sign-agnostic):')
    corr = np.zeros((rk, rk))
    for i in range(rk):
        for j in range(rk):
            c = np.corrcoef(Fb[:, i], Fp[:, j])[0, 1]
            corr[i, j] = c if np.isfinite(c) else 0.0

    used = set()
    best = []
    for i in range(rk):
        order = np.argsort(-np.abs(corr[i]))
        j = next((jj for jj in order if jj not in used), order[0])
        used.add(j)
        best.append((i, j, corr[i, j]))

    for i, j, c in best:
        flag = 'OK' if abs(c) > 0.7 else ('weak' if abs(c) > 0.4 else 'BAD')
        sign = '(inverted)' if c < -0.4 else ''
        print(f'  frozen F{i+1}  <->  proj F{j+1}   r = {c:+.3f}  [{flag}] {sign}')

    median_abs = np.median([abs(c) for _, _, c in best])
    print(f'\nmedian |r| across factors: {median_abs:.3f}')
    if median_abs > 0.7:
        print('  [PASS] Rigorous factors agree with Phase B. The full run '
              'will be comparable to the pragmatic results. Proceed.')
    elif median_abs > 0.4:
        print('  [WARN] Partial agreement. Some factors match, others drift. '
              'Likely a convergence (n_iter_max) or init issue, OR a partial '
              'scale mismatch. Inspect which factors are weak. Factor 4 '
              '(the hypothesis factor) matching well is what matters most.')
    else:
        print('  [FAIL] Rigorous factors do NOT match Phase B. Almost '
              'certainly the tensor score_col is mis-scaled or inverted '
              'relative to Phase B. DO NOT trust a full run yet:')
        print('         1. Ask Giorgos which column Phase B used as values.')
        print('         2. Try score_col = 1 - magnitude_rank (so large = '
              'strong), rebuild, and re-run this check.')
        print('         3. This mismatch is the most likely cause of LR/LDA '
              'collapsing to AUC 0.50 while GNB/XGB survive.')

    # specifically report Factor 4 (hypothesis-relevant)
    if rk >= 4:
        f4 = next((c for i, j, c in best if i == 3), None)
        if f4 is not None:
            print(f'\nHYPOTHESIS FACTOR — frozen Factor 4 best |r| = {abs(f4):.3f} '
                  f'{"(GOOD)" if abs(f4) > 0.7 else "(check this)"}')


if __name__ == '__main__':
    main()