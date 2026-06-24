"""
Repeated Nested Cross-Validation (rnCV) for classifier benchmarking.

This module exposes a single class, RepeatedNestedCV, that orchestrates
the full nested cross-validation procedure:

    R repetitions (default 10) of N-fold outer CV (default 5),
    each outer training fold tuned via K-fold inner CV (default 3)
    using Optuna with TPE sampling (default 50 trials per fold).

The class supports three modes via constructor flags:

    1. Baseline (tune_hyperparams=False):
       Default hyperparameters, no inner loop.

    2. Tuned rnCV (tune_hyperparams=True):
       Full nested CV with Optuna tuning.

    3. Tuned rnCV + feature selection (feature_selector=...):
       Feature selection is the FIRST STEP OF THE PIPELINE, so selection
       is refit on the inner-training split inside every Optuna trial and
       on the full outer-training fold at refit time. Selection is
       therefore fully nested: inner-validation folds never influence the
       selected subset.

Two design points specific to this pipeline:

    - All features are continuous, so the preprocessor does not need to
      know the surviving column layout ahead of time. The selector lives
      inside the pipeline and the preprocessor is built with
      available_cols=None (it adapts to whatever columns the selector
      passes through).

    - Optional external stratification: pass `stratify_labels` to
      fit_evaluate to stratify the outer/inner splits on a composite key
      (e.g. sex x diagnosis) while still training/scoring on the real
      binary `y`. Used for the within-sex value-add design.

Output is a dict mapping each algorithm name to a pandas DataFrame of
shape (R*N, n_metrics), indexed by (round, fold). Plus a summarize()
method that computes median + bootstrap 95% CI per metric per algorithm.
"""

from __future__ import annotations

import time
import warnings
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor
from .metrics import (
    compute_all_metrics, bootstrap_median_ci,
    METRIC_NAMES, INNER_SCORER,
)


# Suppress Optuna's per-trial logging — we have our own progress reporting
optuna.logging.set_verbosity(optuna.logging.WARNING)


class RepeatedNestedCV:
    """
    Repeated Nested Cross-Validation orchestrator.

    Parameters
    ----------
    estimators : dict[str, type | BaseEstimator]
        Dict mapping algorithm name to either a classifier class (when
        tune_hyperparams=True) or an instantiated estimator (when False).

    param_spaces : dict[str, callable], optional
        Dict mapping algorithm name to a function (trial) -> dict of
        hyperparameters. Required when tune_hyperparams=True.

    feature_selector : object, optional
        An UNFITTED sklearn-compatible transformer exposing
        fit(X, y) / transform(X) / get_support() -> list[str]. When
        provided, a fresh deepcopy is inserted as the first step of every
        pipeline, so selection is refit per inner fold (fully nested) and
        per outer-train fold. Use MRMRSelector from feature_selection.py.

    n_rounds : int, default 10
        Number of repetitions (R).
    n_outer : int, default 5
        Outer-loop folds (N).
    n_inner : int, default 3
        Inner-loop folds (K). Ignored when tune_hyperparams=False.
    tune_hyperparams : bool, default True
        If False, runs the outer loop only with default hyperparameters.
    n_trials : int, default 50
        Optuna trials per inner-loop study.

    inner_metric : str, default 'matthews_corrcoef'
        sklearn scorer string for inner-loop optimization.
    outer_metrics : list[str], optional
        Subset of metric names to compute in the outer loop.

    random_state : int, default 42
        Master seed. Per-round seeds are derived as random_state + r.
    n_jobs : int, default -1
        Reserved (inner CV is run manually with deepcopy).
    verbose : bool, default True
        If True, print progress per algorithm and per round.

    Attributes (after fit_evaluate)
    -------------------------------
    results_ : dict[str, pd.DataFrame]
        Per-algorithm metrics. Index = MultiIndex(round, fold).
    selection_log_ : dict[str, list[tuple[int, int, list[str]]]]
        Per-algorithm log of (round, fold, selected_features).
    best_params_log_ : dict[str, list[dict]]
        Per-algorithm log of best hyperparameters per outer fold.
    """

    def __init__(
        self,
        estimators: dict,
        param_spaces: Optional[dict] = None,
        feature_selector=None,
        n_rounds: int = 10,
        n_outer: int = 5,
        n_inner: int = 3,
        tune_hyperparams: bool = True,
        n_trials: int = 50,
        inner_metric: str = INNER_SCORER,
        outer_metrics: Optional[list[str]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True,
    ):
        if tune_hyperparams and param_spaces is None:
            raise ValueError(
                'param_spaces must be provided when tune_hyperparams=True'
            )
        if tune_hyperparams:
            missing = set(estimators) - set(param_spaces or {})
            if missing:
                raise ValueError(
                    f'param_spaces missing entries for: {sorted(missing)}'
                )

        if outer_metrics is not None:
            unknown = set(outer_metrics) - set(METRIC_NAMES)
            if unknown:
                raise ValueError(
                    f'Unknown outer_metrics: {sorted(unknown)}. '
                    f'Available: {METRIC_NAMES}'
                )

        self.estimators = estimators
        self.param_spaces = param_spaces or {}
        self.feature_selector = feature_selector
        self.n_rounds = n_rounds
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.tune_hyperparams = tune_hyperparams
        self.n_trials = n_trials
        self.inner_metric = inner_metric
        self.outer_metrics = outer_metrics or list(METRIC_NAMES)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.results_: dict[str, pd.DataFrame] = {}
        self.selection_log_: dict[str, list] = {}
        self.best_params_log_: dict[str, list] = {}

    # =====================================================================
    # Main entry point
    # =====================================================================

    def fit_evaluate(self, X, y, stratify_labels=None) -> dict[str, pd.DataFrame]:
        """
        Run the full rnCV procedure for every estimator.

        Parameters
        ----------
        X : pandas DataFrame, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
            The real (binary) target used for training and scoring.
        stratify_labels : array-like, shape (n_samples,), optional
            Composite labels used ONLY to stratify the outer and inner
            splits (e.g. f'{sex}_{diagnosis}'). When None, splits are
            stratified on `y` (the original behaviour). Training and
            scoring always use `y`, never these labels.

        Returns
        -------
        results : dict[str, pd.DataFrame]
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be a pandas DataFrame for rnCV.')
        y = np.asarray(y)

        if stratify_labels is None:
            strat = y
        else:
            strat = np.asarray(stratify_labels)
            if strat.shape[0] != y.shape[0]:
                raise ValueError('stratify_labels must align with y.')

        for algo_name in self.estimators:
            if self.verbose:
                mode = 'tuned' if self.tune_hyperparams else 'default'
                fs = ' + FS' if self.feature_selector is not None else ''
                st = ' + ext-strat' if stratify_labels is not None else ''
                print(f'\n[{algo_name}] starting rnCV ({mode}{fs}{st})')

            algo_results, selections, best_params = self._run_single_algorithm(
                algo_name, X, y, strat
            )

            self.results_[algo_name] = algo_results
            self.selection_log_[algo_name] = selections
            self.best_params_log_[algo_name] = best_params

            if self.verbose:
                med = algo_results['MCC'].median()
                print(f'[{algo_name}] done — median MCC = {med:.4f}')

        return self.results_

    # =====================================================================
    # Per-algorithm orchestrator
    # =====================================================================

    def _run_single_algorithm(self, algo_name, X, y, strat):
        """
        Execute R rounds × N outer folds for one algorithm.

        `strat` drives the StratifiedKFold splits; `y` is used for
        training and scoring.
        """
        rows = []
        selections = []
        best_params = []

        t0 = time.time()
        for r in range(self.n_rounds):
            seed = self.random_state + r
            outer = StratifiedKFold(
                n_splits=self.n_outer, shuffle=True, random_state=seed
            )

            # StratifiedKFold splits on `strat`; we index X/y/strat by the
            # returned positions so the inner loop can re-stratify too.
            for fold_i, (train_idx, test_idx) in enumerate(outer.split(X, strat)):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                strat_tr = strat[train_idx]

                fold_metrics, fold_selected, fold_params = self._run_outer_fold(
                    algo_name, X_tr, y_tr, X_te, y_te, strat_tr, seed
                )

                rows.append({'round': r, 'fold': fold_i, **fold_metrics})
                if fold_selected is not None:
                    selections.append((r, fold_i, fold_selected))
                if fold_params is not None:
                    best_params.append({'round': r, 'fold': fold_i, **fold_params})

            if self.verbose:
                elapsed = time.time() - t0
                print(f'  round {r+1}/{self.n_rounds} done '
                      f'(elapsed: {elapsed:.0f}s)')

        results_df = (
            pd.DataFrame(rows)
              .set_index(['round', 'fold'])
              [self.outer_metrics]
        )
        return results_df, selections, best_params

    # =====================================================================
    # Single outer-fold logic
    # =====================================================================

    def _run_outer_fold(self, algo_name, X_tr, y_tr, X_te, y_te, strat_tr, seed):
        """
        Process one outer fold with FULLY NESTED feature selection:
          1. If tuning, run Optuna inner loop on the outer-train fold.
             Each trial builds a pipeline whose FIRST step is a fresh
             feature selector, so selection is refit on each inner-train
             split — inner-val never influences the subset.
          2. Build the final pipeline (selector + preprocessor + classifier)
             with the best params and fit it on the full outer-train fold
             (this refits the selector on all outer-train donors).
          3. Read back the selected features from the fitted selector for
             the stability log.
          4. Evaluate on the outer-test fold.
        """
        # --- Step 1: Hyperparameter tuning (inner loop) ----------------
        if self.tune_hyperparams:
            best_params = self._run_inner_loop(
                algo_name, X_tr, y_tr, strat_tr, seed
            )
        else:
            best_params = None  # use estimator's existing params

        # --- Step 2: Build + fit final pipeline on full outer-train ----
        pipeline = self._build_pipeline(algo_name, best_params)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipeline.fit(X_tr, y_tr)

        # --- Step 3: Read selected features from the fitted selector ---
        selected_features = None
        if self.feature_selector is not None:
            selected_features = list(pipeline.named_steps['selector'].get_support())

        # --- Step 4: Evaluate on outer-test ----------------------------
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            metrics = compute_all_metrics(pipeline, X_te, y_te)
        metrics = {k: metrics[k] for k in self.outer_metrics}

        return metrics, selected_features, best_params

    # =====================================================================
    # Inner Optuna loop
    # =====================================================================

    def _cv_score_deepcopy(self, pipe, X, y, strat, splitter):
        """
        Manual k-fold CV using deepcopy instead of sklearn.clone.

        The split is stratified on `strat`; the pipeline is fit/scored on
        `y`. Because the pipeline's first step is the feature selector,
        each fit() here refits selection on the inner-TRAIN split only,
        and scoring on the inner-VAL split goes through the cached
        transform — so inner-val never shapes the selected subset.
        """
        from sklearn.metrics import get_scorer
        scorer = get_scorer(self.inner_metric)
        scores = []
        for tr_idx, va_idx in splitter.split(X, strat):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            pipe_copy = deepcopy(pipe)
            try:
                pipe_copy.fit(X_tr, y_tr)
                scores.append(float(scorer(pipe_copy, X_va, y_va)))
            except Exception:
                scores.append(float('nan'))
        return scores

    def _run_inner_loop(self, algo_name, X_tr, y_tr, strat_tr, seed):
        """
        Run an Optuna study on the outer-train fold. Returns the full
        params dict (fixed + tuned) used by the best trial.

        Each trial builds a full pipeline (selector + preprocessor +
        classifier) and scores it with inner CV, so feature selection is
        nested inside hyperparameter selection.
        """
        cls = self.estimators[algo_name]
        space_fn = self.param_spaces[algo_name]
        inner_splitter = StratifiedKFold(
            n_splits=self.n_inner, shuffle=True, random_state=seed
        )

        params_by_trial = {}

        def objective(trial):
            params = space_fn(trial)
            params_by_trial[trial.number] = params
            pipe = self._build_pipeline_from_params(algo_name, cls, params)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                scores = self._cv_score_deepcopy(
                    pipe, X_tr, y_tr, strat_tr, inner_splitter
                )
            mean_score = float(np.nanmean(scores))
            if np.isnan(mean_score):
                return -1e9
            return mean_score

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        return params_by_trial[study.best_trial.number]

    # =====================================================================
    # Pipeline construction
    # =====================================================================

    def _prepend_selector(self, steps):
        """
        Prepend a fresh feature-selector step to a list of (name, est)
        pipeline steps, if a feature_selector was configured.

        A deepcopy guarantees each pipeline gets an unfitted selector with
        its own state (critical inside parallel/deepcopy CV).
        """
        if self.feature_selector is not None:
            return [('selector', deepcopy(self.feature_selector))] + steps
        return steps

    def _build_pipeline(self, algo_name, best_params):
        """
        Build a fit-ready Pipeline for the outer-fold refit step.

        Layout: [selector?] -> preprocessor -> classifier.
        The preprocessor is built with available_cols=None because all
        features are continuous and the selector decides the columns at
        fit time.
        """
        if self.tune_hyperparams:
            cls = self.estimators[algo_name]
            return self._build_pipeline_from_params(algo_name, cls, best_params)
        else:
            est = deepcopy(self.estimators[algo_name])
            steps = [
                ('preprocessor', build_preprocessor(algo_name, available_cols=None)),
                ('classifier',   est),
            ]
            return Pipeline(self._prepend_selector(steps))

    def _build_pipeline_from_params(self, algo_name, cls, params):
        """
        Construct a Pipeline given an algorithm class and a params dict.
        Used both inside Optuna trials and for the final outer-fold refit.

        Layout: [selector?] -> preprocessor -> classifier.
        """
        steps = [
            ('preprocessor', build_preprocessor(algo_name, available_cols=None)),
            ('classifier',   cls(**params)),
        ]
        return Pipeline(self._prepend_selector(steps))

    # =====================================================================
    # Result summarization
    # =====================================================================

    def summarize(self, confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Return median + bootstrap CI for each metric, per algorithm.
        """
        if not self.results_:
            raise RuntimeError('Call fit_evaluate before summarize.')

        rows = []
        for algo_name, df in self.results_.items():
            row = {'algorithm': algo_name}
            for metric in self.outer_metrics:
                vals = df[metric].values
                low, high = bootstrap_median_ci(
                    vals, confidence_level=confidence_level
                )
                row[(metric, 'median')] = float(np.nanmedian(vals))
                row[(metric, 'ci_low')] = low
                row[(metric, 'ci_high')] = high
            rows.append(row)

        out = pd.DataFrame(rows).set_index('algorithm')
        out.columns = pd.MultiIndex.from_tuples(
            [c if isinstance(c, tuple) else (c, '') for c in out.columns]
        )
        return out

    # =====================================================================
    # Feature-selection stability
    # =====================================================================

    def get_selection_frequency(
        self,
        algo_name: Optional[str] = None,
        threshold: float = 0.8,
    ) -> pd.DataFrame:
        """
        Return per-feature selection frequency across all (round, fold)
        outer iterations.
        """
        if not self.selection_log_:
            raise RuntimeError(
                'No feature selection log. Was feature_selector provided?'
            )

        if algo_name is None:
            algo_name = next(iter(self.selection_log_))

        log = self.selection_log_[algo_name]
        if not log:
            raise RuntimeError(
                f'No selections recorded for {algo_name!r}. '
                'Was feature_selector set when fit_evaluate was called?'
            )

        n_total = len(log)
        from collections import Counter
        counter = Counter()
        for _, _, feats in log:
            counter.update(feats)

        df = pd.DataFrame([
            {'feature': feat,
             'count': count,
             'frequency': count / n_total,
             f'selected (>={threshold:.0%})': (count / n_total) >= threshold}
            for feat, count in counter.most_common()
        ]).set_index('feature')

        return df

    def get_best_hyperparams(self, algo_name: str) -> pd.DataFrame:
        """Return per-fold best Optuna hyperparameters for one algorithm."""
        if algo_name not in self.best_params_log_:
            raise KeyError(f'No record for algorithm {algo_name!r}')
        log = self.best_params_log_[algo_name]
        if not log:
            return pd.DataFrame()
        return pd.DataFrame(log).set_index(['round', 'fold'])