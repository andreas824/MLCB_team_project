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
                 redundancy: str = 'c', n_jobs: int = -1,
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