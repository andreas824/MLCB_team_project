"""
Pseudobulk expression features for donor-level MDD classification.

Two feature constructions are provided:

    per_celltype_pseudobulk(...)   -> the hypothesis-focused baseline:
        one expression vector per (donor, cell type) for a chosen set of
        cell types (default: the microglia -> ExN10_L46 pair), concatenated
        side by side with cell-type-prefixed column names. This isolates the
        expression signal in exactly the cell types the communication
        hypothesis names, so it compares directly against the Factor-4
        program.

    whole_donor_pseudobulk(...)    -> the pooled baseline / ablation:
        one expression vector per donor, summing across all nuclei.

Both sum raw counts per group, then apply CPM + log1p. Summation and the
per-row CPM + log1p are per-donor (or per donor-celltype) operations that
do not learn anything from the cohort, so computing them once globally is
leakage-safe -- unlike HVG selection or scaling, which stay inside the CV
folds.

Cell types with no nuclei in a given donor yield an all-zero row for that
donor-celltype block (which becomes log1p(0) = 0 after CPM). Donors that
are entirely missing a cell type therefore contribute a zero vector for
that block; the downstream imputation/selection handles those.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


# Default focal cell types: the sender/receiver of the communication
# hypothesis. 'Mic' equals 'Mic1' in this annotation (microglia have a
# single fine cluster), so the sender node is just 'Mic'.
HYPOTHESIS_CELLTYPES = ('Mic', 'ExN10_L46')


def _logcpm_from_counts(counts: np.ndarray) -> np.ndarray:
    """
    CPM + log1p on a (groups x genes) matrix of summed raw counts.

    Each row is divided by its own library size (row sum) and scaled to
    counts-per-million, then log1p-transformed. Rows with zero library
    size (a group with no nuclei) are left as all-zero.
    """
    counts = np.asarray(counts, dtype=float)
    lib = counts.sum(axis=1, keepdims=True)
    safe = np.where(lib > 0, lib, 1.0)          # avoid divide-by-zero
    cpm = counts / safe * 1e6
    cpm[lib.ravel() == 0] = 0.0                 # keep empty groups all-zero
    return np.log1p(cpm)


def _sum_counts_by_group(adata, group_codes, n_groups, layer='counts'):
    """
    Sum raw counts within each group via a sparse indicator matmul.

    group_codes : int array, length n_obs, value in [0, n_groups) or -1 to
                  exclude that nucleus from every group.
    Returns a dense (n_groups x n_genes) array of summed counts.
    """
    keep = group_codes >= 0
    rows = group_codes[keep]
    cols = np.nonzero(keep)[0]
    G = sp.csr_matrix(
        (np.ones(keep.sum(), dtype=float), (rows, cols)),
        shape=(n_groups, adata.n_obs),
    )
    counts = adata.layers[layer]
    summed = G @ counts
    return np.asarray(summed.todense()) if sp.issparse(summed) else np.asarray(summed)


def per_celltype_pseudobulk(
    adata,
    celltypes=HYPOTHESIS_CELLTYPES,
    celltype_col='cc_label',
    donor_col='donor_id',
    layer='counts',
):
    """
    Per-(donor, cell type) pseudobulk logCPM for a set of focal cell types.

    Parameters
    ----------
    adata : AnnData
        Must have raw counts in adata.layers[layer], a donor column and a
        cell-type column in .obs.
    celltypes : sequence of str
        Cell-type labels to build blocks for (default: Mic + ExN10_L46).
    celltype_col, donor_col : str
        Column names in adata.obs.
    layer : str
        Layer holding raw counts.

    Returns
    -------
    pd.DataFrame
        Index = donor_id (sorted, one row per donor present in the data).
        Columns = '<celltype>__<gene>' for every focal cell type x gene.
        A donor with no nuclei of a given cell type has zeros in that block.
    """
    obs = adata.obs
    genes = adata.var_names.astype(str).to_numpy()
    donors = np.sort(obs[donor_col].astype(str).unique())
    donor_to_row = {d: i for i, d in enumerate(donors)}

    donor_str = obs[donor_col].astype(str).to_numpy()
    ct_str = obs[celltype_col].astype(str).to_numpy()

    blocks = []
    for ct in celltypes:
        # group code = donor row index for nuclei of this cell type, else -1
        in_ct = (ct_str == ct)
        codes = np.full(adata.n_obs, -1, dtype=np.int64)
        codes[in_ct] = np.array([donor_to_row[d] for d in donor_str[in_ct]],
                                 dtype=np.int64)
        summed = _sum_counts_by_group(adata, codes, len(donors), layer=layer)
        logcpm = _logcpm_from_counts(summed)
        block = pd.DataFrame(
            logcpm, index=donors,
            columns=[f'{ct}__{g}' for g in genes],
        )
        blocks.append(block)

    out = pd.concat(blocks, axis=1)
    out.index.name = donor_col
    return out


def whole_donor_pseudobulk(
    adata,
    donor_col='donor_id',
    layer='counts',
):
    """
    Whole-donor pseudobulk logCPM (sum across all nuclei per donor).

    Returns
    -------
    pd.DataFrame
        Index = donor_id (sorted). Columns = gene names.
    """
    obs = adata.obs
    genes = adata.var_names.astype(str).to_numpy()
    donors = np.sort(obs[donor_col].astype(str).unique())
    donor_to_row = {d: i for i, d in enumerate(donors)}

    codes = np.array([donor_to_row[d] for d in obs[donor_col].astype(str)],
                     dtype=np.int64)
    summed = _sum_counts_by_group(adata, codes, len(donors), layer=layer)
    logcpm = _logcpm_from_counts(summed)

    out = pd.DataFrame(logcpm, index=donors, columns=genes)
    out.index.name = donor_col
    return out