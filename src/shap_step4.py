#!/usr/bin/env python
"""
Phase C Step 4 — is the microglia->ExN10_L46 communication program (the
hypothesis 'Factor 4') important for MDD, and is it carried in BOTH sexes?

Sex is perfectly confounded with cohort/batch here, so we cannot test
cross-sex generalisation. Instead we ask whether the hypothesis factor
discriminates MDD *within* each sex (a shared-axis prediction).

Two complementary views, both on the frozen Phase B donor factors
(phaseB_donor_factors.parquet), which carry the canonical Factor-4
interpretation:

  1. DIRECT (model-free, robust): univariate discrimination of MDD by the
     hypothesis factor alone — directed AUC + Mann-Whitney U — overall and
     within each sex.
  2. LR linear-SHAP (multivariate): importance of each factor in the
     predictive logistic-regression model (the only communication-only model
     above chance: rigorous CV AUC ~0.633; the tree models were ~0.5, so
     TreeSHAP would be uninformative). Mean |SHAP| per factor, split by sex.

Run:
    python src/shap_step4.py --checkpoint-dir data/checkpoints
"""
from __future__ import annotations
import os, sys, argparse, warnings
warnings.filterwarnings('ignore')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir', default='data/checkpoints')
    ap.add_argument('--fig-dir', default='data/checkpoints/phaseC_results_rigorous',
                    help='where to write the SHAP beeswarm PNGs')
    ap.add_argument('--pseudobulk',
                    default='data/checkpoints/phaseC_pseudobulk_percelltype.parquet',
                    help='per-celltype pseudobulk cache (for the combined-space SHAP)')
    ap.add_argument('--rank', type=int, default=5)
    ap.add_argument('--n-var-prefilter', type=int, default=1000,
                    help='variance pre-filter width before mRMR (mirrors pipeline)')
    ap.add_argument('--k-features', type=int, default=15,
                    help='mRMR k for the combined-space model (mirrors pipeline)')
    ap.add_argument('--n-rounds', type=int, default=3)
    ap.add_argument('--n-outer', type=int, default=5)
    ap.add_argument('--n-iter-max', type=int, default=50,
                    help='CP iterations for the in-fold projector')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-heavy', action='store_true',
                    help='run only sections 1-3 (skip combined + in-fold SHAP)')
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import shap

    cdir = args.checkpoint_dir
    F = pd.read_parquet(os.path.join(cdir, 'phaseB_donor_factors.parquet'))
    F.index = F.index.astype(str)
    factors = list(F.columns)                       # ['Factor 1', ... 'Factor 5']
    senders = pd.read_parquet(os.path.join(cdir, 'phaseB_factor_senders.parquet'))
    receivers = pd.read_parquet(os.path.join(cdir, 'phaseB_factor_receivers.parquet'))
    obs = pd.read_parquet(os.path.join(cdir, 'phaseA_obs.parquet'))
    meta = obs.drop_duplicates('donor_id').set_index('donor_id')[['condition', 'sex']]
    meta.index = meta.index.astype(str)

    df = F.join(meta).dropna(subset=['condition', 'sex'])
    y = (df['condition'] == 'MDD').astype(int).values
    sex = df['sex'].astype(str).values
    print(f'donors: {len(df)}  | MDD={int(y.sum())} Control={int((1-y).sum())}  | '
          f'female={int((sex=="female").sum())} male={int((sex=="male").sum())}')

    # --- identify the hypothesis factor: max Mic-sender x ExN10_L46-receiver ---
    load = (senders.loc['Mic'].astype(float) *
            receivers.loc['ExN10_L46'].astype(float))
    hyp = load.idxmax()
    print(f'\nhypothesis factor (max Mic_sender x ExN10_L46_receiver loading): '
          f'{hyp}  (loading {load[hyp]:.3f})')
    print('  per-factor Mic x ExN10_L46 loading product:')
    for f in factors:
        print(f'    {f}: {load[f]:+.3f}' + ('   <-- hypothesis' if f == hyp else ''))

    # =====================================================================
    # 1. DIRECT univariate discrimination of MDD by the hypothesis factor
    # =====================================================================
    print('\n' + '=' * 68)
    print(f'1. DIRECT — {hyp} alone discriminating MDD (directed AUC, MWU p)')
    print('=' * 68)

    def directed(vals, yy):
        if len(np.unique(yy)) < 2:
            return float('nan'), float('nan')
        auc = roc_auc_score(yy, vals)
        try:
            p = mannwhitneyu(vals[yy == 1], vals[yy == 0]).pvalue
        except ValueError:
            p = float('nan')
        return auc, p

    v = df[hyp].astype(float).values
    for grp, mask in [('overall', np.ones(len(df), bool)),
                      ('female', sex == 'female'),
                      ('male', sex == 'male')]:
        auc, p = directed(v[mask], y[mask])
        disc = max(auc, 1 - auc) if auc == auc else float('nan')
        hi = '(higher in MDD)' if auc >= 0.5 else '(higher in Control)'
        print(f'  {grp:7s} n={int(mask.sum()):2d}  AUC={auc:.3f} '
              f'[discr {disc:.3f}] {hi:18s}  MWU p={p:.3f}')

    # =====================================================================
    # 2. LR linear-SHAP — multivariate importance of each factor, per sex
    # =====================================================================
    print('\n' + '=' * 68)
    print('2. LR linear-SHAP — mean |SHAP| per factor (log-odds), by sex')
    print('=' * 68)
    X = df[factors].astype(float).values
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(penalty='l2', C=1.0, max_iter=5000).fit(Xs, y)
    print(f'  (in-sample LR AUC={roc_auc_score(y, lr.predict_proba(Xs)[:,1]):.3f}; '
          f'rigorous CV reference AUC~0.633)')

    explainer = shap.LinearExplainer(lr, Xs)
    sv = explainer.shap_values(Xs)
    sv = np.asarray(sv)
    if sv.ndim == 3:                      # (n, k, classes) -> positive class
        sv = sv[:, :, -1]

    def mean_abs(mask):
        return np.abs(sv[mask]).mean(axis=0)

    rows = {'overall': mean_abs(np.ones(len(df), bool)),
            'female': mean_abs(sex == 'female'),
            'male': mean_abs(sex == 'male')}
    hdr = '  ' + 'group'.ljust(8) + ''.join(f'{f.replace(" ",""):>10s}' for f in factors)
    print(hdr)
    for g, vals in rows.items():
        print('  ' + g.ljust(8) + ''.join(f'{x:10.3f}' for x in vals))

    # rank of hypothesis factor by global importance
    order = np.argsort(-rows['overall'])
    rank = list(np.array(factors)[order]).index(hyp) + 1
    print(f'\n  {hyp} global mean|SHAP| rank: {rank}/{len(factors)}  '
          f'(1 = most important)')
    hi = factors.index(hyp)
    print(f'  {hyp} mean|SHAP|  female={rows["female"][hi]:.3f}  '
          f'male={rows["male"][hi]:.3f}  '
          f'-> {"comparable across sexes" if min(rows["female"][hi],rows["male"][hi])/max(rows["female"][hi],rows["male"][hi]+1e-9) > 0.5 else "asymmetric across sexes"}')

    # =====================================================================
    # 3. SHAP beeswarm plots (overall + per sex) — distribution of each
    #    factor's signed log-odds contribution across donors. Colour = factor
    #    value (red high / blue low), so you can read directionality, not just
    #    magnitude: e.g. high Factor 4 pushing toward MDD shows as red dots on
    #    the positive-SHAP side.
    # =====================================================================
    print('\n' + '=' * 68)
    print('3. SHAP beeswarm plots')
    print('=' * 68)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.fig_dir, exist_ok=True)
    short = [f.replace(' ', '') for f in factors]            # 'Factor4'
    hi_lab = short[factors.index(hyp)]

    def beeswarm(sv_mat, Xmat, feat_names, hi_label, mask, yvec, tag, title,
                 prefix='step4_shap_beeswarm'):
        """Reusable beeswarm panel. `mask` selects rows; `yvec` is the label
        vector aligned to those rows (for the MDD count in the title)."""
        n = int(mask.sum())
        if n < 3 or len(np.unique(yvec[mask])) < 2:
            print(f'  {tag:8s} skipped (n={n}, needs >=3 donors and both classes)')
            return
        plt.figure()
        shap.summary_plot(sv_mat[mask], Xmat[mask], feature_names=feat_names,
                          show=False, sort=True, plot_size=(7, 3.5))
        ax = plt.gca()
        for t in ax.get_yticklabels():       # highlight the hypothesis factor
            if t.get_text() == hi_label:
                t.set_color('crimson'); t.set_fontweight('bold')
        ax.set_title(f'{title}  (n={n}, MDD={int(yvec[mask].sum())})  '
                     f'— red label = hypothesis {hyp}', fontsize=9)
        out = os.path.join(args.fig_dir, f'{prefix}_{tag}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  wrote {out}')

    beeswarm(sv, Xs, short, hi_lab, np.ones(len(df), bool), y,
             'overall', 'LR linear-SHAP — all donors')
    beeswarm(sv, Xs, short, hi_lab, sex == 'female', y,
             'female', 'LR linear-SHAP — female donors')
    beeswarm(sv, Xs, short, hi_lab, sex == 'male', y,
             'male', 'LR linear-SHAP — male donors')

    if args.skip_heavy:
        return

    # =====================================================================
    # 4. COMBINED-SPACE SHAP — does Factor 4 survive against expression?
    #    Comm factors and expression genes compete in ONE model. Mirrors the
    #    rigorous pipeline's feature path: variance-top-N genes + the 5 comm
    #    factors -> mRMR(k) over all of them together -> LR -> SHAP. If
    #    Factor 4 ranks high here it carries signal NOT redundant with
    #    expression; if it is dropped by selection, the communication signal
    #    is an echo of expression.
    # =====================================================================
    print('\n' + '=' * 68)
    print('4. COMBINED-SPACE SHAP — comm factors + expression genes, one model')
    print('=' * 68)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # src/
    from feature_selection import MRMRSelector

    pb = pd.read_parquet(args.pseudobulk)
    pb.index = pb.index.astype(str)
    genes = pb.loc[df.index]                       # align to labelled donors
    v = genes.var(axis=0)
    top_genes = list(v.sort_values(ascending=False).index[:args.n_var_prefilter])
    facs_named = df[factors].copy()
    facs_named.columns = short                     # 'Factor 4' -> 'Factor4'
    combX = pd.concat([facs_named, genes[top_genes]], axis=1)
    print(f'  candidate features: {combX.shape[1]} '
          f'({len(short)} comm factors + {len(top_genes)} variance-top genes)')

    sel = MRMRSelector(k=args.k_features, relevance='mi').fit(combX, y)
    feats = list(sel.selected_features_)
    n_fac_sel = sum(f in short for f in feats)
    print(f'  mRMR selected k={len(feats)}: {n_fac_sel} comm factor(s) + '
          f'{len(feats) - n_fac_sel} gene(s)')
    surv = hi_lab in feats
    print(f'  hypothesis {hyp} survived selection vs expression: '
          f'{"YES" if surv else "NO  -> redundant with expression at this k"}')

    Xc = StandardScaler().fit_transform(combX[feats].values)
    lrc = LogisticRegression(penalty='l2', C=1.0, max_iter=5000).fit(Xc, y)
    print(f'  (in-sample combined LR AUC='
          f'{roc_auc_score(y, lrc.predict_proba(Xc)[:,1]):.3f})')
    svc = np.asarray(shap.LinearExplainer(lrc, Xc).shap_values(Xc))
    if svc.ndim == 3:
        svc = svc[:, :, -1]
    mabs_c = np.abs(svc).mean(axis=0)
    order_c = np.argsort(-mabs_c)
    print('  mean|SHAP| ranking (combined model):')
    for rk, j in enumerate(order_c, 1):
        kind = ('  <-- HYPOTHESIS' if feats[j] == hi_lab else
                '   (comm factor)' if feats[j] in short else '')
        print(f'    {rk:2d}. {feats[j]:26s} {mabs_c[j]:.3f}{kind}')
    if surv:
        rank_c = list(np.array(feats)[order_c]).index(hi_lab) + 1
        print(f'  -> {hyp} ranks {rank_c}/{len(feats)} among comm+expression '
              f'features')
    beeswarm(svc, Xc, feats, hi_lab, np.ones(len(df), bool), y, 'overall',
             'Combined SHAP — comm factors + expression genes',
             prefix='step4_shap_beeswarm_combined')

    # =====================================================================
    # 5. IN-FOLD (leakage-safe) SHAP — factors refit inside each CV fold.
    #    Sections 1-4 use the all-donor frozen factors (mild leakage: test
    #    donors helped shape the factor space). Here the CP is refit on each
    #    outer-TRAIN fold only and donors are projected onto those patterns;
    #    SHAP is computed on the HELD-OUT donors. We reuse the rigorous run's
    #    exact outer splits (StratifiedKFold, seed+r). Non-negative CP has no
    #    sign ambiguity, only permutation, so each fold's components are
    #    matched to the canonical Factor 1..5 by |corr| of TRAIN-donor
    #    loadings (Hungarian assignment). This is the honest counterpart to
    #    section 2 and should agree with the rigorous CV AUC (~0.633).
    # =====================================================================
    print('\n' + '=' * 68)
    print('5. IN-FOLD SHAP — factors refit per outer-train fold (no leakage)')
    print('=' * 68)
    from sklearn.model_selection import StratifiedKFold
    from scipy.optimize import linear_sum_assignment
    from tensor_features import TensorFactorProjector, _cp_cache_clear
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    import run_rigorous_local as RR

    print('  building 4D tensor from per-donor LIANA (one-time) ...', flush=True)
    liana = pd.read_parquet(os.path.join(cdir, 'phaseB_liana_per_donor.parquet'))
    T, donor_order = RR.build_tensor(liana)

    donors_all = [d for d in donor_order if d in df.index]   # tensor order
    meta_all = df.loc[donors_all]
    y_all = (meta_all['condition'] == 'MDD').astype(int).values
    sex_all = meta_all['sex'].astype(str).values
    strat_all = (meta_all['sex'].astype(str) + '_' +
                 meta_all['condition'].astype(str)).values
    Ffroz = meta_all[factors].values                         # canonical, aligned
    X_id = pd.DataFrame(index=pd.Index(donors_all, name='donor_id'))
    n = len(donors_all)
    hi_idx = factors.index(hyp)
    _cp_cache_clear()

    pooled_sv, pooled_x, pooled_y, pooled_sex = [], [], [], []
    aucs = []
    for r in range(args.n_rounds):
        skf = StratifiedKFold(n_splits=args.n_outer, shuffle=True,
                              random_state=args.seed + r)
        for tr_idx, te_idx in skf.split(np.zeros(n), strat_all):
            proj = TensorFactorProjector(tensor=T, donor_order=donor_order,
                                         rank=args.rank, random_state=0,
                                         n_iter_max=args.n_iter_max)
            proj.fit(X_id.iloc[tr_idx])
            load_all = proj.transform(X_id).values            # (n, rank), in-fold cols
            # match in-fold columns -> canonical via |corr| on TRAIN donors only
            A, Bc = load_all[tr_idx], Ffroz[tr_idx]
            Cm = np.zeros((args.rank, args.rank))
            for i in range(args.rank):
                for j in range(args.rank):
                    if A[:, i].std() > 1e-12 and Bc[:, j].std() > 1e-12:
                        Cm[i, j] = abs(np.corrcoef(A[:, i], Bc[:, j])[0, 1])
            row, colm = linear_sum_assignment(-Cm)            # in-fold i -> canonical c
            inf2can = {i: c for i, c in zip(row, colm)}

            sc = StandardScaler().fit(load_all[tr_idx])
            Xtr, Xte = sc.transform(load_all[tr_idx]), sc.transform(load_all[te_idx])
            lr_if = LogisticRegression(penalty='l2', C=1.0,
                                       max_iter=5000).fit(Xtr, y_all[tr_idx])
            if len(np.unique(y_all[te_idx])) == 2:
                aucs.append(roc_auc_score(y_all[te_idx],
                                          lr_if.predict_proba(Xte)[:, 1]))
            sv_te = np.asarray(shap.LinearExplainer(lr_if, Xtr).shap_values(Xte))
            if sv_te.ndim == 3:
                sv_te = sv_te[:, :, -1]
            # reorder in-fold columns into canonical Factor 1..rank order
            sv_can = np.zeros_like(sv_te)
            x_can = np.zeros_like(Xte)
            for i_inf, c in inf2can.items():
                sv_can[:, c] = sv_te[:, i_inf]
                x_can[:, c] = Xte[:, i_inf]
            pooled_sv.append(sv_can); pooled_x.append(x_can)
            pooled_y.append(y_all[te_idx]); pooled_sex.append(sex_all[te_idx])

    pooled_sv = np.vstack(pooled_sv); pooled_x = np.vstack(pooled_x)
    pooled_y = np.concatenate(pooled_y); pooled_sex = np.concatenate(pooled_sex)
    print(f'  held-out LR AUC across {len(aucs)} folds: '
          f'mean={np.mean(aucs):.3f}  (rigorous CV reference ~0.633)')

    def mabs_if(mask):
        return np.abs(pooled_sv[mask]).mean(axis=0)
    rows_if = {'overall': mabs_if(np.ones(len(pooled_sv), bool)),
               'female': mabs_if(pooled_sex == 'female'),
               'male': mabs_if(pooled_sex == 'male')}
    print('  mean|SHAP| per canonical factor (held-out donors), by sex:')
    print('  ' + 'group'.ljust(8) + ''.join(f'{s:>10s}' for s in short))
    for g, vals in rows_if.items():
        print('  ' + g.ljust(8) + ''.join(f'{x:10.3f}' for x in vals))
    order_if = np.argsort(-rows_if['overall'])
    rank_if = list(np.array(factors)[order_if]).index(hyp) + 1
    print(f'\n  {hyp} in-fold mean|SHAP| rank: {rank_if}/{len(factors)}  '
          f'(section 2 frozen-factor rank was {rank})')
    print(f'  {hyp} mean|SHAP|  female={rows_if["female"][hi_idx]:.3f}  '
          f'male={rows_if["male"][hi_idx]:.3f}')
    beeswarm(pooled_sv, pooled_x, short, hi_lab,
             np.ones(len(pooled_sv), bool), pooled_y, 'overall',
             'In-fold SHAP — leakage-safe factors (held-out donors)',
             prefix='step4_shap_beeswarm_infold')


if __name__ == '__main__':
    main()
