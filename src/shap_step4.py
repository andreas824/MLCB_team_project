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

    def beeswarm(mask, tag, title):
        n = int(mask.sum())
        if n < 3 or len(np.unique(y[mask])) < 2:
            print(f'  {tag:8s} skipped (n={n}, needs >=3 donors and both classes)')
            return
        plt.figure()
        # order='importance' default; keep all 5 factors so the panel is comparable
        shap.summary_plot(sv[mask], Xs[mask], feature_names=short,
                          show=False, sort=True, plot_size=(7, 3.5))
        ax = plt.gca()
        # highlight the hypothesis factor tick label
        for t in ax.get_yticklabels():
            if t.get_text() == hi_lab:
                t.set_color('crimson'); t.set_fontweight('bold')
        ax.set_title(f'{title}  (n={n}, MDD={int(y[mask].sum())})  '
                     f'— red label = hypothesis {hyp}', fontsize=9)
        out = os.path.join(args.fig_dir, f'step4_shap_beeswarm_{tag}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  wrote {out}')

    beeswarm(np.ones(len(df), bool), 'overall', 'LR linear-SHAP — all donors')
    beeswarm(sex == 'female', 'female', 'LR linear-SHAP — female donors')
    beeswarm(sex == 'male', 'male', 'LR linear-SHAP — male donors')


if __name__ == '__main__':
    main()
