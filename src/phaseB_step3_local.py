#!/usr/bin/env python
"""Phase B Step 3, local runner (e.g. Apple Silicon M3).

Runs the Tensor-cell2cell decomposition off the per-donor LIANA parquet, so you
can do the elbow + decomposition uninterrupted on a laptop instead of fighting
Colab disconnects. Needs ONLY the small checkpoints, not the 5.8 GB matrix:
    - phaseB_liana_per_donor.parquet   (produced by notebook Step 2)
    - phaseA_obs.parquet               (produced by notebook Step 0)

Outputs (written next to the inputs, in --checkpoint-dir):
    - phaseB_factor_{contexts,senders,receivers,lrs}.parquet
    - phaseB_donor_factors.parquet     (donor x factor -> Phase C features)
    - phaseB_rank.json                 (only if --elbow)

Usage:
    python src/phaseB_step3_local.py --checkpoint-dir data/checkpoints
    python src/phaseB_step3_local.py --checkpoint-dir data/checkpoints --elbow
    python src/phaseB_step3_local.py --checkpoint-dir data/checkpoints --rank 8

Env (Python >= 3.10):
    conda create -n mlcb python=3.11 -y && conda activate mlcb
    pip install -r requirements-local.txt
"""
import os
import json
import argparse
import warnings

warnings.filterwarnings('ignore')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint-dir', required=True,
                    help='folder holding phaseB_liana_per_donor.parquet + phaseA_obs.parquet')
    ap.add_argument('--rank', type=int, default=None,
                    help='fixed rank; overrides phaseB_rank.json and --default-rank')
    ap.add_argument('--default-rank', type=int, default=10)
    ap.add_argument('--elbow', action='store_true',
                    help='run elbow rank selection first and write phaseB_rank.json')
    ap.add_argument('--upper-rank', type=int, default=10)
    args = ap.parse_args()

    import pandas as pd
    import liana as li
    from scipy.stats import mannwhitneyu

    cdir = args.checkpoint_dir
    res = pd.read_parquet(os.path.join(cdir, 'phaseB_liana_per_donor.parquet'))
    print('liana_res:', res.shape, '| donors:', res['donor_id'].nunique())

    # --- 4D tensor straight from the per-donor results -----------------------
    tensor = li.multi.to_tensor_c2c(
        liana_res=res, sample_key='donor_id',
        score_key='magnitude_rank', how='outer_cells',
    )
    print('tensor shape (donor, sender, receiver, LR):', tensor.shape)

    # --- rank: --rank > elbow/json > default ---------------------------------
    rank_file = os.path.join(cdir, 'phaseB_rank.json')
    if args.elbow:
        tensor.elbow_rank_selection(upper_rank=args.upper_rank, runs=1,
                                    init='random', automatic_elbow=True, random_state=0)
        print('elbow suggested rank:', tensor.rank)
        json.dump({'rank': int(tensor.rank)}, open(rank_file, 'w'))

    if args.rank is not None:
        rank = args.rank
    elif os.path.exists(rank_file):
        rank = json.load(open(rank_file))['rank']
    else:
        rank = args.default_rank
    print('decomposing at rank =', rank)

    # --- single decomposition (random init: safe for the masked tensor) ------
    tensor.compute_tensor_factorization(rank=rank, init='random', random_state=0)

    # --- save factor matrices ------------------------------------------------
    factors = tensor.factors
    print('factor dims:', list(factors.keys()))
    namemap = {'Contexts': 'contexts', 'Sender Cells': 'senders',
               'Receiver Cells': 'receivers', 'Ligand-Receptor Pairs': 'lrs'}
    for dim, fac in factors.items():
        short = namemap.get(dim, dim.lower().replace(' ', '_').replace('-', '_'))
        fac.to_parquet(os.path.join(cdir, f'phaseB_factor_{short}.parquet'))
    ctx = factors.get('Contexts', list(factors.values())[0])
    ctx.to_parquet(os.path.join(cdir, 'phaseB_donor_factors.parquet'))
    print('donor x factor loadings:', ctx.shape, '-> saved to', cdir)

    # --- quick confound read-out (same logic as notebook Step 3b) ------------
    obs = pd.read_parquet(os.path.join(cdir, 'phaseA_obs.parquet'))
    meta = obs.drop_duplicates('donor_id').set_index('donor_id')[['condition', 'sex']]
    senders, receivers = factors.get('Sender Cells'), factors.get('Receiver Cells')
    if (senders is not None and 'Mic' in senders.index and
            receivers is not None and 'ExN10_L46' in receivers.index):
        hypo = (senders.loc['Mic'].astype(float)
                * receivers.loc['ExN10_L46'].astype(float)).sort_values(ascending=False)
        hf = hypo.index[0]
        df = ctx.join(meta)
        pc = mannwhitneyu(df.loc[df.condition == 'MDD', hf],
                          df.loc[df.condition == 'Control', hf])[1]
        ps = mannwhitneyu(df.loc[df.sex == 'female', hf],
                          df.loc[df.sex == 'male', hf])[1]
        print(f'\nhypothesis factor = {hf}  '
              f'(Mic x ExN10_L46 loading {hypo.iloc[0]:.3f})')
        print(f'  vs diagnosis:  p = {pc:.4f}')
        print(f'  vs sex/cohort: p = {ps:.4f}')


if __name__ == '__main__':
    main()
