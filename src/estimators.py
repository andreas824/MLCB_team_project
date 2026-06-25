"""
Classifier registry: defaults, hyperparameter search spaces, and
construction utilities for the algorithms compared at the donor level.

Two main entry points are exposed:

    get_default_estimators()       -> baseline mode (no tuning)
    get_estimators_and_spaces()    -> Optuna tuning mode

Each algorithm has:
    - A class reference, so the rnCV class can instantiate fresh copies.
    - A function `<algo>_space(trial)` returning a dict of suggested
      hyperparameters for one Optuna trial.
    - A function `<algo>_fixed_params()` returning parameters that are
      NOT tuned (random_state, solver, etc.).

Hyperparameter ranges are chosen for the small donor-level sample
(n = 71, ~57 donors per outer-train fold) and the mild class imbalance.
Tree depths and ensemble sizes are kept conservative: with so few
training donors per fold, deep or large ensembles memorize noise. The
ranges stay wide enough for Optuna to find heavier regularization but
narrow enough to avoid clearly-overfit regions.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# Master random seed. Fixed for reproducibility.
RANDOM_STATE = 42


# --- Logistic Regression (Elastic Net) ----------------------------------

def lr_fixed_params() -> dict:
    """
    Non-tuned parameters for LR with elastic-net regularization.

    saga is the only scikit-learn solver that supports elastic-net.
    max_iter is high because saga is iterative and can be slow to
    converge on small standardized datasets. class_weight='balanced'
    handles the mild class imbalance.

    l1_ratio defaults to 0.5 (equal L1/L2 blend) so baseline mode -- where
    Optuna does not supply l1_ratio -- has a valid configuration. In
    tuning mode, lr_space() overrides it with a sampled value.
    """
    return {
        'penalty': 'elasticnet',
        'solver': 'saga',
        'max_iter': 5000,
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'l1_ratio': 0.5,
    }


def lr_space(trial) -> dict:
    """
    LR Optuna space.

    C        : log-uniform [1e-3, 1e2] -- five orders of magnitude. Small
               sample, so a wide range lets Optuna find heavy regularization.
    l1_ratio : uniform [0, 1] -- trades off lasso (feature selection) vs
               ridge (smooth shrinkage). Useful with correlated genes.
    """
    return {
        **lr_fixed_params(),
        'C':        trial.suggest_float('C', 1e-3, 1e2, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
    }


# --- Gaussian Naive Bayes -----------------------------------------------

def gnb_fixed_params() -> dict:
    """GNB has no fixed parameters worth setting."""
    return {}


def gnb_space(trial) -> dict:
    """
    GNB has a single tunable parameter -- the variance smoothing constant.
    Default is 1e-9. The range covers six orders of magnitude around it.
    Serves as a cheap independence-assuming baseline: if richer models do
    not beat it, the signal is weak.
    """
    return {
        'var_smoothing': trial.suggest_float('var_smoothing', 1e-12, 1e-6, log=True),
    }


# --- Linear Discriminant Analysis ---------------------------------------

def lda_fixed_params() -> dict:
    """LDA has no parameters that should be fixed across configurations."""
    return {}


def lda_space(trial) -> dict:
    """
    LDA Optuna space.

    solver    : {'svd', 'lsqr', 'eigen'}. svd is the default. lsqr/eigen
                support shrinkage, which stabilizes the covariance estimate
                on small samples with correlated features.
    shrinkage : conditional on solver. Valid only for lsqr/eigen, in [0,1].
                0 -> no regularization; 1 -> diagonal covariance. Left as
                None for svd to avoid sklearn errors.
    """
    solver = trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen'])
    params = {'solver': solver}
    if solver in ('lsqr', 'eigen'):
        params['shrinkage'] = trial.suggest_float('shrinkage', 0.0, 1.0)
    return params


# --- Random Forest ------------------------------------------------------

def rf_fixed_params() -> dict:
    return {
        'n_jobs': 1,
        'random_state': RANDOM_STATE,
        'class_weight': 'balanced',
    }


def rf_space(trial) -> dict:
    """
    RF Optuna space, sized for ~57 training donors per fold.

    n_estimators     : [100, 300] -- a few hundred trees saturate accuracy
                       at this sample size; more only adds runtime.
    max_depth        : [2, 6] -- primary regularizer. Shallow trees are
                       essential with so few donors per fold.
    min_samples_leaf : [2, 12] -- leaves below ~2-3 donors are statistically
                       meaningless here.
    max_features     : {'sqrt', 'log2', 0.5} -- fraction of features per split.
    """
    return {
        **rf_fixed_params(),
        'n_estimators':     trial.suggest_int('n_estimators', 100, 300, step=50),
        'max_depth':        trial.suggest_int('max_depth', 2, 6),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 12),
        'max_features':     trial.suggest_categorical('max_features',
                                                      ['sqrt', 'log2', 0.5]),
    }


# --- XGBoost ------------------------------------------------------------

def xgb_fixed_params() -> dict:
    return {
        'objective':    'binary:logistic',
        'eval_metric':  'logloss',
        'verbosity':    0,
        'random_state': RANDOM_STATE,
        'n_jobs':       1,
    }


def xgb_space(trial) -> dict:
    """
    XGBoost Optuna space, sized for ~57 training donors per fold.

    Depth-wise tree growth, so max_depth is the main capacity knob and is
    kept shallow. gamma and the L1/L2 leaf penalties add regularization;
    subsample / colsample inject randomness against overfitting.

    n_estimators     : [100, 600] -- boosting tolerates more (weak) trees
                       than RF, but kept moderate for the small sample.
    learning_rate    : log [1e-2, 0.3] -- classic interaction with n_estimators.
    max_depth        : [2, 5] -- shallow; deep trees memorize at this size.
    min_child_weight : [1, 10] -- larger values are more conservative.
    gamma            : [0, 5] -- minimum loss reduction to split (pruning).
    reg_alpha        : log [1e-8, 1.0] -- L1 leaf regularization.
    reg_lambda       : log [1e-8, 1.0] -- L2 leaf regularization.
    subsample        : [0.6, 1.0] -- row sub-sampling per tree.
    colsample_bytree : [0.6, 1.0] -- feature sub-sampling per tree.
    """
    return {
        **xgb_fixed_params(),
        'n_estimators':     trial.suggest_int('n_estimators', 100, 600, step=50),
        'learning_rate':    trial.suggest_float('learning_rate', 1e-2, 0.3, log=True),
        'max_depth':        trial.suggest_int('max_depth', 2, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma':            trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }


# --- Quadratic Discriminant Analysis (OPT-IN, not in default registry) --
# QDA estimates a separate covariance per class, which is unstable at this
# sample size unless heavily regularized. It is NOT included in the default
# registry. Add it only if non-linear models (XGB/RF) clearly beat the
# linear ones, signalling non-linear structure worth modelling. When added,
# reg_param shrinks each class covariance toward the diagonal -- the only
# safe form of QDA for small n. To enable: add 'QDA' entries to the dicts
# in get_estimators_and_spaces / get_default_estimators using the functions
# below.

def qda_fixed_params() -> dict:
    """QDA has no parameters that should be fixed across configurations."""
    return {}


def qda_space(trial) -> dict:
    """
    QDA Optuna space.

    reg_param : [0, 1] -- shrinks each class covariance toward the diagonal.
                Near 0 is full (unstable) QDA; higher values regularize
                heavily, which is what makes QDA viable on a small sample.
    """
    return {
        'reg_param': trial.suggest_float('reg_param', 0.0, 1.0),
    }


# --- Public API ---------------------------------------------------------

def get_estimators_and_spaces() -> tuple[dict, dict]:
    """
    Return parallel dicts of estimator classes and their Optuna spaces.
    Used by RepeatedNestedCV in tuning mode.

    Returns
    -------
    estimators : dict[str, type]
    spaces : dict[str, callable]   # each is (trial) -> dict[str, Any]
    """
    estimators = {
        'LR':  LogisticRegression,
        'GNB': GaussianNB,
        'LDA': LinearDiscriminantAnalysis,
        'RF':  RandomForestClassifier,
        'XGB': XGBClassifier,
    }
    spaces = {
        'LR':  lr_space,
        'GNB': gnb_space,
        'LDA': lda_space,
        'RF':  rf_space,
        'XGB': xgb_space,
    }
    return estimators, spaces


def get_default_estimators() -> dict:
    """
    Return ready-to-use estimator instances with default hyperparameters
    (plus our fixed_params: random_state and minor settings).
    Used by RepeatedNestedCV in baseline mode (no inner loop, no tuning).
    """
    return {
        'LR':  LogisticRegression(**lr_fixed_params()),
        'GNB': GaussianNB(**gnb_fixed_params()),
        'LDA': LinearDiscriminantAnalysis(**lda_fixed_params()),
        'RF':  RandomForestClassifier(**rf_fixed_params()),
        'XGB': XGBClassifier(**xgb_fixed_params()),
    }