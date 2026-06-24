"""
Preprocessing builders for the MDD case/control classification pipeline.

The Phase C feature space is entirely continuous:
    - pseudobulk log-CPM gene expression (per donor)
    - the communication factor loadings from the tensor decomposition

There are no categorical, ordinal, or binary features. Two estimator
families are supported:

    - 'linear' : LR / ElasticNet-LR, GNB, LDA   -> impute + scale
    - 'tree'   : RF, XGBoost, (LightGBM)         -> impute only (trees
                 are invariant to monotonic feature transforms)

The mapping from estimator name to family lives in `_FAMILY_MAP`. The
public entry point is `build_preprocessor(name, available_cols)`.

All preprocessors are built fresh on each call so the returned
ColumnTransformer is UNFITTED -- essential when it is fitted inside a
cross-validation loop on training data only.

Imputation note
---------------
Pseudobulk expression never produces NaNs (it is a sum). The
communication factors CAN be NaN for donors whose focal cell types fell
below the min-cell threshold and were masked in the tensor. Median
imputation is kept as a safety net so those donors don't crash the
linear models; fitted inside each fold, it uses train-donor medians only.
"""

from __future__ import annotations

from typing import Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# --- Estimator-family routing -------------------------------------------
# Adding a new estimator only requires editing this map.

_FAMILY_MAP = {
    'LR':   'linear',
    'GNB':  'linear',
    'LDA':  'linear',
    'RF':   'tree',
    'LGBM': 'tree',
    'XGB':  'tree',
}


# --- Public API ----------------------------------------------------------

def build_preprocessor(estimator_name: str, available_cols: Optional[list] = None):
    """
    Return an unfitted preprocessor appropriate for `estimator_name`.

    Parameters
    ----------
    estimator_name : str
        One of the keys in `_FAMILY_MAP`.
    available_cols : list of str, optional
        The columns to transform. When None, the transformer matches every
        column it is fitted on (used when an upstream selector decides the
        columns at fit time). When a list is given, the ColumnTransformer
        names its output predictably, which keeps SHAP feature names clean.

    Returns
    -------
    BaseEstimator
        A fresh, unfitted sklearn-compatible transformer. All features are
        treated as continuous.
    """
    if estimator_name not in _FAMILY_MAP:
        raise ValueError(
            f'Unknown estimator: {estimator_name!r}. Known: {list(_FAMILY_MAP)}'
        )

    family = _FAMILY_MAP[estimator_name]

    if family == 'linear':
        cont_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
        ])
    elif family == 'tree':
        cont_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
        ])
    else:
        raise RuntimeError(f'No builder implemented for family {family!r}')

    if available_cols is not None:
        return ColumnTransformer(
            transformers=[('cont', cont_pipe, list(available_cols))],
            remainder='drop',
        )
    else:
        from sklearn.compose import make_column_selector
        return ColumnTransformer(
            transformers=[('cont', cont_pipe, make_column_selector())],
            remainder='drop',
        )


def get_estimator_family(estimator_name: str) -> str:
    """Return the preprocessing family ('linear' or 'tree') for an estimator."""
    if estimator_name not in _FAMILY_MAP:
        raise ValueError(
            f'Unknown estimator: {estimator_name!r}. Known: {list(_FAMILY_MAP)}'
        )
    return _FAMILY_MAP[estimator_name]