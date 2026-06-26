"""
In-fold communication-factor features via tensor factorization.

This module exposes TensorFactorProjector, an sklearn-compatible transformer
that makes the Phase B communication factors leakage-safe inside cross-
validation. Instead of using the factors precomputed on all donors (which
lets test donors influence the factor space), it:

    .fit(X_train)   -> refits a non-negative CP decomposition on the
                       sub-tensor of the TRAINING donors only, learning the
                       ligand-receptor / sender / receiver factor patterns
                       without ever seeing the test donors.
    .transform(X)    -> projects each donor in X onto those fixed patterns
                       via non-negative least squares (NNLS), recovering its
                       per-factor loadings.

Because it lives as the first step of a pipeline, the rnCV machinery fits it
on the training fold and only transforms the test fold -- so test donors
never shape the factors. This is the rigorous counterpart to using the
frozen phaseB_donor_factors.parquet (the pragmatic path).

Design
------
The full 4D tensor T (donor x LR x sender x receiver) is expensive to build
(minutes, from the 13M-row per-donor LIANA table), so it is built ONCE
outside the transformer and passed in at construction together with the
donor-order mapping. The transformer keys on donor IDs read from the index
of X: X carries only donor identity (one row per donor, index = donor_id),
not the factors themselves -- those are recomputed each fold.

The CP decomposition is non-negative (matching Tensor-cell2cell), fit with
tensorly. Masked tensor entries (NaN, where a donor had too few cells of a
cell type) are handled by a binary weight mask in the decomposition and are
dropped from the NNLS projection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.optimize import nnls

import tensorly as tl
from tensorly.decomposition import non_negative_parafac


# Module-level cache of CP decompositions, keyed by the exact set of train
# donors (plus rank/seed). The rnCV machinery deepcopies the projector for
# every pipeline, so an instance attribute would not be shared -- but inside
# one outer fold every inner trial refits on the SAME train donors. Caching
# here collapses those ~60 identical refits per outer fold into one, the
# single biggest speedup for the rigorous run. The cache holds only a handful
# of entries (one per distinct train split) so memory is not a concern.
_CP_CACHE: dict = {}


def _cp_cache_clear():
    _CP_CACHE.clear()


class TensorFactorProjector(BaseEstimator, TransformerMixin):
    """
    Leakage-safe communication factors via in-fold tensor factorization.

    Parameters
    ----------
    tensor : np.ndarray, shape (n_donors, d1, d2, d3)
        The full 4D communication tensor (donor x LR x sender x receiver),
        with np.nan where a donor-celltype slice was masked. Built once and
        shared across all transformer instances (read-only).
    donor_order : list[str]
        Donor IDs in the order of axis 0 of `tensor`.
    rank : int, default 5
        CP rank (number of communication programs).
    random_state : int, default 0
        Seed for the non-negative CP decomposition.
    n_iter_max : int, default 200
        Max iterations for the CP solver.

    Attributes (after fit)
    ----------------------
    factors_ : list[np.ndarray]
        The fitted CP factor matrices [A_donor, B_lr, C_sender, D_receiver].
        Only B, C, D are used for projecting new donors; A is the train
        donors' own loadings.
    design_ : np.ndarray, shape (d1*d2*d3, rank)
        Precomputed design matrix for NNLS projection.
    feature_names_out_ : list[str]
        ['Factor_1', ..., 'Factor_rank'].
    """

    def __init__(self, tensor, donor_order, rank: int = 5,
                 random_state: int = 0, n_iter_max: int = 50,
                 tol: float = 1e-6, use_cache: bool = True):
        self.tensor = tensor
        self.donor_order = donor_order
        self.rank = rank
        self.random_state = random_state
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.use_cache = use_cache
        self._pos_of = {d: i for i, d in enumerate(donor_order)}

    # -- helpers ---------------------------------------------------------

    def _positions(self, X):
        """Map the donor IDs in X's index to tensor axis-0 positions."""
        if isinstance(X, pd.DataFrame):
            ids = [str(d) for d in X.index]
        else:
            # if X is array-like we assume it already holds positions
            return list(np.asarray(X).ravel().astype(int))
        missing = [d for d in ids if d not in self._pos_of]
        if missing:
            raise KeyError(f'donor IDs not in tensor: {missing[:5]}')
        return [self._pos_of[d] for d in ids]

    def _build_design(self):
        """
        Design matrix M where column r is the outer product
        vec(B[:,r] x C[:,r] x D[:,r]) in C-order, matching T[i].reshape(-1).
        Lets us solve T[i] ~ M @ loadings for each donor via NNLS.
        """
        _, B, C, D = self.factors_
        R = B.shape[1]
        M = np.empty((B.shape[0] * C.shape[0] * D.shape[0], R))
        for r in range(R):
            M[:, r] = np.einsum('i,j,k->ijk', B[:, r], C[:, r], D[:, r]).reshape(-1)
        return M

    # -- sklearn API -----------------------------------------------------

    def fit(self, X, y=None):
        train_pos = self._positions(X)

        # cache key: exact train-donor set (order-independent) + rank + seed.
        # Within one outer fold, all inner trials hit the same key, so the
        # expensive CP runs once instead of ~60 times.
        key = (frozenset(train_pos), self.rank, self.random_state,
               self.n_iter_max)
        if self.use_cache and key in _CP_CACHE:
            self.factors_ = _CP_CACHE[key]
        else:
            sub = self.tensor[train_pos].astype(float)    # (n_tr, d1, d2, d3)
            mask = (~np.isnan(sub)).astype(float)         # 1 obs, 0 masked
            sub_filled = np.nan_to_num(sub, nan=0.0)
            tl.set_backend('numpy')
            weights, factors = non_negative_parafac(
                tl.tensor(sub_filled),
                rank=self.rank,
                mask=tl.tensor(mask),
                init='random',
                random_state=self.random_state,
                n_iter_max=self.n_iter_max,
                tol=self.tol,
            )
            self.factors_ = [np.asarray(f) for f in factors]
            if self.use_cache:
                _CP_CACHE[key] = self.factors_

        self.design_ = self._build_design()
        self.feature_names_out_ = [f'Factor_{r+1}' for r in range(self.rank)]
        return self

    def transform(self, X):
        pos = self._positions(X)
        M = self.design_
        out = np.zeros((len(pos), self.rank))
        for k, p in enumerate(pos):
            s = self.tensor[p].reshape(-1)
            m = np.isfinite(s)
            # NNLS on observed entries only (drop masked)
            out[k], _ = nnls(M[m], s[m])
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(out, index=X.index,
                                columns=self.feature_names_out_)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)