"""
Model-agnostic feature selection for the rnCV pipeline.

Exposes one class, MRMRSelector: an sklearn-compatible transformer wrapping
the mRMR (minimum Redundancy Maximum Relevance) algorithm. It slots in as
the first step of a pipeline so selection is fit on training data only and
applied to held-out data via transform -- the requirement for leakage-free
cross-validation.

Relevance options
-----------------
mRMR scores each feature by relevance to the target and penalizes
redundancy with already-selected features. The relevance statistic is
configurable:

    'mi' : mutual information (mutual_info_classif). Captures non-linear
           feature-target dependence, not just linear/monotonic. This is
           the default here: the communication factors and gene programs
           may relate to diagnosis non-linearly.
    'f'  : ANOVA F-statistic. Fast, but only senses linear separation.
    'ks' : Kolmogorov-Smirnov statistic.
    'rf' : random-forest importance.

Redundancy is Pearson correlation between features ('c').

Note on 'mi': the underlying mrmr_selection package does not ship a
mutual-information relevance out of the box, so we supply it as a callable
(MI is exactly the relevance term in Peng et al.'s original mRMR, so this
is faithful to the method, not a workaround for its own sake).
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif

from mrmr import mrmr_classif


# --- Mutual-information relevance, matching mrmr's callable contract -----
# mrmr calls relevance_func(X=DataFrame, y=Series) and expects a Series
# (index = feature name, value = relevance score). mutual_info_classif is
# stochastic via its k-NN estimator, so we fix random_state for
# reproducible selection across folds.

def _mi_relevance(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.Series:
    """Mutual information between every column of X and y, as a Series."""
    mi = mutual_info_classif(
        X.values, np.asarray(y),
        discrete_features=False,
        random_state=random_state,
    )
    return pd.Series(mi, index=X.columns).fillna(0.0)


class MRMRSelector(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible wrapper around mrmr_classif.

    Parameters
    ----------
    k : int
        Number of features to select.
    relevance : {'mi', 'f', 'ks', 'rf'}, default 'mi'
        Feature-target relevance statistic. 'mi' uses mutual information
        (non-linear aware); the rest are passed through to mrmr_selection.
    redundancy : {'c'}, default 'c'
        Redundancy measure: 'c' = Pearson correlation between features.
    n_jobs : int, default -1
        Parallelism for the built-in relevance statistics.
    random_state : int, default 42
        Seed for the mutual-information estimator (only used when
        relevance='mi'), so selection is reproducible across folds.

    Attributes (after fit)
    ----------------------
    selected_features_ : list[str]
        Selected column names, in mRMR ranking order.
    feature_names_in_ : list[str]
        Original column names seen during fit.
    """

    def __init__(self, k: int = 7, relevance: str = 'mi',
                 redundancy: str = 'c', n_jobs: int = 1,
                 random_state: int = 42):
        self.k = k
        self.relevance = relevance
        self.redundancy = redundancy
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _relevance_arg(self):
        """
        Resolve the `relevance` parameter to what mrmr_classif expects:
        a callable for 'mi', otherwise the string it already understands.
        """
        if self.relevance == 'mi':
            return functools.partial(_mi_relevance, random_state=self.random_state)
        return self.relevance

    def fit(self, X, y):
        """
        Run mRMR on (X, y) and cache the selected feature names.

        X should be a pandas DataFrame so names are preserved. A numpy
        array is wrapped with positional names ('f0', 'f1', ...).
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])

        self.feature_names_in_ = list(X.columns)
        k_eff = min(self.k, X.shape[1])

        # mrmr requires X.index == y.index. Inside CV, X is a sliced frame
        # while y is a numpy array; reset both to a fresh RangeIndex.
        X_aligned = X.reset_index(drop=True)
        y_aligned = pd.Series(np.asarray(y)).reset_index(drop=True)

        self.selected_features_ = mrmr_classif(
            X=X_aligned, y=y_aligned,
            K=k_eff,
            relevance=self._relevance_arg(),
            redundancy=self.redundancy,
            n_jobs=self.n_jobs,
            show_progress=False,
        )
        return self

    def transform(self, X):
        """
        Subset X to the columns selected during fit.

        Returns a DataFrame (preserving names) when X is a DataFrame,
        otherwise a numpy array.
        """
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        col_to_idx = {name: i for i, name in enumerate(self.feature_names_in_)}
        idx = [col_to_idx[f] for f in self.selected_features_]
        return X[:, idx]

    def get_support(self) -> list[str]:
        """Return the names of selected features, in mRMR rank order."""
        return list(self.selected_features_)


# --- Variance pre-filter ------------------------------------------------
# With tens of thousands of pseudobulk genes, running mRMR (and especially
# mutual-information relevance) over the full feature set is intractable
# inside nested CV. A variance pre-filter keeps the top-N most variable
# features before mRMR sees them. Fitted on the training split only (it
# lives in the pipeline ahead of the selector), so it never looks at
# held-out data -- the same leakage discipline as every other fitted step.

class VarianceTopK(BaseEstimator, TransformerMixin):
    """
    Keep the top-N features by training-set variance, plus any protected
    features that must always pass through.

    A fast, unsupervised first-stage filter that shrinks a very wide
    feature matrix to a tractable width before mRMR. Unsupervised (variance
    only, never the label), so it is a mild operation; still fitted in-fold
    for strict leakage safety.

    Protected features bypass the variance ranking entirely and are always
    retained. This matters when feature sets are on different scales: e.g.
    pseudobulk log-CPM genes have far larger variance than the communication
    factor loadings, so without protection a plain variance filter would
    silently drop every factor from a combined gene+factor matrix.

    Parameters
    ----------
    n_top : int, default 1000
        Number of highest-variance NON-protected features to keep. If fewer
        non-protected columns exist, all are kept.
    protect : sequence of str or None, default None
        Column names that must always be retained, regardless of variance.
        Matched by name (requires a DataFrame). They are kept in addition
        to the n_top variance-ranked features.
    protect_prefixes : sequence of str or None, default None
        Convenience: any column whose name starts with one of these
        prefixes is protected. Useful for protecting a whole block (e.g.
        all 'Factor_' columns) without listing each name.

    Attributes (after fit)
    ----------------------
    selected_features_ : list[str]
        Names of the retained columns (protected + top variance), in
        original column order.
    """

    def __init__(self, n_top: int = 1000, protect=None, protect_prefixes=None):
        self.n_top = n_top
        self.protect = protect
        self.protect_prefixes = protect_prefixes

    def _is_protected(self, name: str) -> bool:
        if self.protect is not None and name in set(self.protect):
            return True
        if self.protect_prefixes is not None:
            return any(str(name).startswith(p) for p in self.protect_prefixes)
        return False

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            variances = X.var(axis=0, ddof=1).values
        else:
            self.feature_names_in_ = [f'f{i}' for i in range(X.shape[1])]
            variances = np.var(np.asarray(X), axis=0, ddof=1)

        names = self.feature_names_in_
        protected_idx = [i for i, nm in enumerate(names) if self._is_protected(nm)]
        protected_set = set(protected_idx)

        # rank only the non-protected features by variance
        candidate_idx = np.array([i for i in range(len(names))
                                  if i not in protected_set])
        n_keep = min(self.n_top, len(candidate_idx))
        if len(candidate_idx) > 0:
            order = candidate_idx[np.argsort(variances[candidate_idx])[::-1]]
            top_idx = list(order[:n_keep])
        else:
            top_idx = []

        keep = sorted(set(top_idx) | protected_set)  # union, original order
        self.support_idx_ = np.array(keep, dtype=int)
        self.selected_features_ = [names[i] for i in keep]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        return np.asarray(X)[:, self.support_idx_]

    def get_support(self) -> list[str]:
        return list(self.selected_features_)