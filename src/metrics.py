"""
Evaluation metrics and statistical summaries for the rnCV pipeline.

This module exposes three things:

1. METRIC_NAMES — canonical list of metrics computed at every outer fold.
2. compute_all_metrics(pipeline, X_test, y_test) — returns a dict with
   every metric for one outer-fold evaluation.
3. bootstrap_median_ci(values, ...) — non-parametric 95% CI for the
   median, computed via percentile bootstrap.

Design notes
------------
- specificity is computed manually from the confusion matrix because
  scikit-learn does not expose a `specificity_score` directly.
- The probability-based metrics (AUC, PR-AUC) require predict_proba.
  If a classifier doesn't expose it, those metrics return NaN rather
  than crashing — the rest of the row is still usable.
- The inner-loop metric (used by Optuna) is referenced by its scikit-learn
  scorer string; it lives separately so it can be passed directly to
  cross_val_score without going through compute_all_metrics.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    matthews_corrcoef,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    average_precision_score,
    confusion_matrix,
)
from scipy.stats import bootstrap


# Canonical metric ordering — used for DataFrame column order
METRIC_NAMES = [
    'MCC',
    'AUC',
    'BA',
    'F1',
    'Recall',
    'Specificity',
    'Precision',
    'PR-AUC',
]


# --- Per-fold metric computation ----------------------------------------

def compute_all_metrics(pipeline, X_test, y_test) -> dict[str, float]:
    """
    Compute all classification metrics for one outer-fold evaluation.

    Parameters
    ----------
    pipeline : fitted sklearn-compatible Pipeline or estimator
        Must implement .predict(X). predict_proba is optional; AUC and
        PR-AUC will be NaN if absent.
    X_test, y_test : array-like
        Held-out test fold features and labels.

    Returns
    -------
    dict
        Mapping from metric name (in METRIC_NAMES) to scalar score.
        AUC/PR-AUC are NaN if the pipeline does not expose predict_proba.
    """
    y_pred = pipeline.predict(X_test)

    # Probability-based metrics: only if predict_proba is available
    if hasattr(pipeline, 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        auc    = float(roc_auc_score(y_test, y_proba))
        pr_auc = float(average_precision_score(y_test, y_proba))
    else:
        auc, pr_auc = float('nan'), float('nan')

    # Confusion matrix for specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn) / float(tn + fp) if (tn + fp) > 0 else float('nan')

    return {
        'MCC':         float(matthews_corrcoef(y_test, y_pred)),
        'AUC':         auc,
        'BA':          float(balanced_accuracy_score(y_test, y_pred)),
        'F1':          float(f1_score(y_test, y_pred, zero_division=0)),
        'Recall':      float(recall_score(y_test, y_pred, zero_division=0)),
        'Specificity': specificity,
        'Precision':   float(precision_score(y_test, y_pred, zero_division=0)),
        'PR-AUC':      pr_auc,
}

# --- Bootstrap confidence interval --------------------------------------

def bootstrap_median_ci(
    values,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Non-parametric percentile bootstrap CI for the median.

    Parameters
    ----------
    values : 1-D array-like
        Sample of metric values (typically 50, from 10 rounds × 5 folds).
    confidence_level : float, default 0.95
        Two-sided coverage. 0.95 yields the 2.5%/97.5% percentiles.
    n_resamples : int, default 10_000
        Bootstrap iterations. Standard value; CI width stabilizes well
        before 10k.
    random_state : int, default 42
        Seed for reproducibility.

    Returns
    -------
    (low, high) : tuple of float
        Lower and upper bounds of the CI. Returns (nan, nan) if input
        contains only NaNs.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (np.nan, np.nan)

    res = bootstrap(
        (arr,),
        statistic=np.median,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method='percentile',
        random_state=random_state,
    )
    return (float(res.confidence_interval.low),
            float(res.confidence_interval.high))


# --- Inner-loop scorer name (for Optuna / cross_val_score) --------------

INNER_SCORER = 'matthews_corrcoef'
"""
Scoring string passed to sklearn.model_selection.cross_val_score inside
the Optuna objective. MCC is preferred over AUC for imbalanced binary
classification (Chicco & Jurman, 2020) because it weights all four
confusion-matrix entries symmetrically.
"""