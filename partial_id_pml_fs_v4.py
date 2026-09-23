"""
Partial-identification feature selection for partial multi-label data.

Input MAT variables (configurable):
    data            : n_samples x n_features
    partial_labels  : n_labels x n_samples or n_samples x n_labels
    target          : n_labels x n_samples or n_samples x n_labels

The script reports the same seven downstream metrics as robust.py, using
one-vs-rest LinearSVC models trained on partial_labels.  It produces two
selection tracks:
    1. Necessary: all features certified as necessary at a feature-relative
       tolerance epsilon = eta * (F0* - F*), where F0* is the best
       intercept-only objective.
    2. RankedProtocol: necessity-first ranking, then ||W[k, :]||_2, evaluated
       at top 1--20% of the original dimension.  Water is evaluated at
       feature counts 1,...,d, matching the paper's stated special protocol.

By default a cumulative group-deletion diagnostic is run on the final
all-sample model.  Exponential search followed by binary refinement finds the
smallest top-ranked prefix certified as jointly necessary within the requested
range.  Use ``--no-group-diagnostic`` to skip it.

Version 4 additionally performs a held-out matched-removal validation at
5%, 10%, and 20% feature budgets.  For every evaluation fold it compares the
original top-k subset with (i) the same-size subset after removing the
training-certified necessary set or jointly necessary prefix and refilling
from lower ranks, and (ii) ten size-matched random-removal controls.  Fold-level
feature identities and pairwise Jaccard stability are saved for auditing.

Edit DATASET_PATHS near the bottom, or use --data on the command line.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import f1_score, hamming_loss, label_ranking_loss
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


FLOATX = np.float32
EPS = 1e-12
METRIC_COLUMNS = [
    "RankingLoss",
    "HammingLoss",
    "CoverageError",
    "OneError",
    "AveragePrecision",
    "MicroF1",
    "MacroF1",
]
LOWER_IS_BETTER_METRICS = {
    "RankingLoss",
    "HammingLoss",
    "CoverageError",
    "OneError",
}


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class SelectorConfig:
    # feature_gain: epsilon = eta * (F0* - F*), which compares a deletion cost
    # with the total objective improvement attributable to all features.
    # total_objective retains the v2 definition epsilon = eta * |F*|.
    epsilon_mode: str = "feature_gain"  # feature_gain | total_objective
    epsilon_ratio: float = 0.01
    necessity_mode: str = "auto"  # auto | fast | exact

    max_iter: int = 140
    tuning_max_iter: int = 55
    off_max_iter: int = 90
    min_iter: int = 12
    tol: float = 2e-5
    power_iterations: int = 12
    intercept_iterations: int = 80
    intercept_reference_max_iter: int = 3000
    certificate_refine_iterations: int = 30

    # A feature-level statement at tolerance epsilon is only numerically
    # trustworthy when the optimizer's computable suboptimality certificate is
    # small relative to epsilon.  Final and constrained fits are automatically
    # continued until this target is met or the refinement budget is exhausted.
    certificate_gap_ratio_target: float = 0.05
    adaptive_refinement_rounds: int = 3
    # Refinement is triggered only when the computable certificate is still
    # too large.  A sizeable continuation block is important for weakly
    # regularized, ill-conditioned fits; already-converged fits pay no cost.
    adaptive_refinement_max_iter: int = 1000
    adaptive_tolerance_shrink: float = 0.1
    adaptive_min_tol: float = 1e-9

    lambda_ratio_grid: Sequence[float] = (1e-3, 1e-2, 1e-1)
    gamma_ratio_grid: Sequence[float] = (1e-4, 1e-3, 1e-2)
    boundary_expansion_rounds: int = 3
    lambda_ratio_expansion_max: float = 1.0

    # Final full-data diagnostic for the extension from singleton necessity to
    # jointly necessary nested groups.  Most tested groups are rejected by
    # cheap bounds; exact optimization is used only when a bound straddles the
    # tolerance interval.
    enable_group_diagnostic: bool = True
    group_diagnostic_only_when_no_singleton: bool = False
    group_diagnostic_max_fraction: float = 0.20
    group_diagnostic_max_features: int = 256
    group_diagnostic_stop_after_first_necessary: bool = True
    group_diagnostic_binary_refine: bool = True

    # In auto mode, exact refits are skipped for very costly n*d*q problems.
    auto_skip_exact_ndq: float = 2e8
    auto_max_exact_refits: int = 20
    max_zero_bound_evaluations: int = 100

    optimization_slack_floor: float = 1e-12
    random_state: int = 42
    verbose: bool = True


@dataclass
class EvaluationConfig:
    n_splits: int = 5
    tuning_fold_index: int = 0
    tuning_metric: str = "MicroF1"
    svm_c: float = 1.0
    svm_tol: float = 1e-3
    svm_max_iter: int = 50000
    svm_n_jobs: int = 1
    random_state: int = 42
    fit_full_after_cv: bool = True
    water_feature_count_protocol: bool = True
    enable_matched_removal: bool = True
    matched_removal_budgets: Sequence[int] = (5, 10, 20)
    matched_removal_random_repeats: int = 10
    enable_fold_group_diagnostic: bool = True
    # Sparse-like dense MAT matrices become extremely slow if centering turns
    # their zeros into nonzeros.  Below this density, keep zeros and use CSR for
    # LinearSVC while retaining variance scaling and an intercept.
    preserve_zeros_density_threshold: float = 0.20
    svm_sparse_density_threshold: float = 0.20


@dataclass
class FitResult:
    W: np.ndarray
    b: np.ndarray
    objective: float
    smooth_objective: float
    stationarity_residual: float
    optimization_gap_upper: float
    n_iter: int
    elapsed_seconds: float
    optimization_gap_ratio: float = float("nan")
    certificate_converged: bool = False
    adaptive_refinement_rounds: int = 0


@dataclass
class InterceptOnlyFit:
    b: np.ndarray
    objective: float
    optimization_gap_upper: float
    n_iter: int
    elapsed_seconds: float
    certificate_converged: bool


@dataclass
class EpsilonReference:
    mode: str
    epsilon: float
    epsilon_lower: float
    epsilon_upper: float
    feature_gain: float
    feature_gain_lower: float
    feature_gain_upper: float
    intercept_objective: float
    intercept_optimization_gap_upper: float
    intercept_certificate_converged: bool
    intercept_n_iter: int
    intercept_elapsed_seconds: float


@dataclass
class NecessityResult:
    necessary: np.ndarray
    unresolved: np.ndarray
    off_gap_lower: np.ndarray
    off_gap_upper: np.ndarray
    off_gap_estimate: np.ndarray
    epsilon: float
    epsilon_lower: float
    epsilon_upper: float
    epsilon_mode: str
    feature_gain: float
    feature_gain_lower: float
    feature_gain_upper: float
    intercept_objective: float
    intercept_optimization_gap_upper: float
    intercept_certificate_converged: bool
    intercept_n_iter: int
    intercept_elapsed_seconds: float
    exact_refits: int
    exact_refits_converged: int
    max_off_optimization_gap_ratio: float
    mode_used: str
    elapsed_seconds: float


@dataclass(frozen=True)
class BudgetSpec:
    index: int
    feature_ratio: float
    num_features: int
    protocol: str


@dataclass
class GroupDiagnosticResult:
    group_size: int
    feature_indices: np.ndarray
    group_weight_norm: float
    gap_lower: float
    gap_upper: float
    gap_estimate: float
    epsilon: float
    epsilon_lower: float
    epsilon_upper: float
    status: str
    exact_refit: bool
    certificate_converged: bool
    optimization_gap_ratio: float
    sum_singleton_lower: float
    sum_singleton_upper: float
    certified_synergy_lower: float
    elapsed_seconds: float


# -----------------------------------------------------------------------------
# MAT loading and validation
# -----------------------------------------------------------------------------


def _orient_labels(Y: np.ndarray, n_samples: int, name: str) -> np.ndarray:
    Y = np.asarray(Y)
    if Y.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {Y.shape}.")
    if Y.shape[0] == n_samples:
        out = Y
    elif Y.shape[1] == n_samples:
        out = Y.T
    else:
        raise ValueError(
            f"Cannot align {name} shape {Y.shape} with data sample count {n_samples}."
        )
    return (np.asarray(out) > 0).astype(np.int8, copy=False)


def load_mat_dataset(
    filepath: str,
    x_key: str = "data",
    partial_key: str = "partial_labels",
    target_key: str = "target",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(filepath)
    if path.suffix.lower() != ".mat":
        raise ValueError(f"Expected a .mat file, got: {filepath}")
    if not path.exists():
        raise FileNotFoundError(path)

    values = loadmat(path, variable_names=[x_key, partial_key, target_key])
    missing = [key for key in (x_key, partial_key, target_key) if key not in values]
    if missing:
        raise KeyError(f"MAT file {path} is missing variables: {missing}")

    X = np.asarray(values[x_key], dtype=FLOATX)
    if X.ndim != 2:
        raise ValueError(f"{x_key} must be 2-D, got shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{x_key} contains NaN or infinite values.")

    Y_partial = _orient_labels(values[partial_key], X.shape[0], partial_key)
    Y_target = _orient_labels(values[target_key], X.shape[0], target_key)
    if Y_partial.shape != Y_target.shape:
        raise ValueError(
            f"partial_labels shape {Y_partial.shape} and target shape "
            f"{Y_target.shape} do not match after orientation."
        )

    violations = int(np.sum((Y_target > 0) & (Y_partial == 0)))
    if violations:
        warnings.warn(
            f"Found {violations} target positives outside partial_labels; "
            "the usual partial-label containment assumption is violated.",
            RuntimeWarning,
        )
    return X, Y_partial, Y_target


def summarize_dataset(name: str, X: np.ndarray, Y: np.ndarray, Y_t: np.ndarray) -> str:
    return (
        f"[{name}] X={X.shape}, labels={Y.shape[1]}, "
        f"partial positives/sample={Y.sum(axis=1).mean():.3f}, "
        f"target positives/sample={Y_t.sum(axis=1).mean():.3f}, "
        f"empty partial rows={int(np.sum(Y.sum(axis=1) == 0))}"
    )


# -----------------------------------------------------------------------------
# Candidate-label convex hull and ambiguity loss
# -----------------------------------------------------------------------------


def project_candidate_hull(P: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    """Project rows onto {z: 0<=z<=s, sum(z)>=1}.

    For nonempty candidate sets, clipping is sufficient when the clipped sum
    is at least one.  Otherwise the solution is the probability-simplex
    projection restricted to candidate coordinates.  Empty candidate rows are
    projected to the all-zero vector.
    """
    P = np.asarray(P, dtype=FLOATX)
    mask = np.asarray(candidate_mask, dtype=bool)
    Q = np.clip(P, 0.0, 1.0)
    Q *= mask

    counts = mask.sum(axis=1)
    sums = Q.sum(axis=1, dtype=np.float64)
    need = (counts > 0) & (sums < 1.0 - 1e-7)
    if not np.any(need):
        return Q.astype(FLOATX, copy=False)

    V = P[need]
    M = mask[need]
    m = counts[need]
    # Non-candidates sort after all finite candidate coordinates.
    U = np.sort(np.where(M, V, -np.inf), axis=1)[:, ::-1]
    U_finite = np.where(np.isfinite(U), U, 0.0)
    cssv = np.cumsum(U_finite, axis=1, dtype=np.float64)
    j = np.arange(1, P.shape[1] + 1, dtype=np.float64)[None, :]
    valid_pos = j <= m[:, None]
    cond = valid_pos & (U - (cssv - 1.0) / j > 0.0)
    rho = np.maximum(cond.sum(axis=1) - 1, 0)
    theta = (cssv[np.arange(len(rho)), rho] - 1.0) / (rho + 1.0)
    Q_need = np.maximum(V - theta[:, None], 0.0) * M
    Q[need] = Q_need.astype(FLOATX, copy=False)
    return Q.astype(FLOATX, copy=False)


def ambiguity_loss_from_predictions(
    P: np.ndarray, candidate_mask: np.ndarray
) -> Tuple[float, np.ndarray]:
    Q = project_candidate_hull(P, candidate_mask)
    residual = (P - Q).astype(FLOATX, copy=False)
    loss = 0.5 * float(np.sum(residual * residual, dtype=np.float64)) / P.shape[0]
    return loss, residual


# -----------------------------------------------------------------------------
# Scale estimation and convex proximal optimization
# -----------------------------------------------------------------------------


def estimate_design_lipschitz(
    X: np.ndarray, n_iter: int = 12, random_state: int = 42
) -> float:
    """Estimate ||[X,1]||_2^2 / n by power iteration."""
    n, d = X.shape
    rng = np.random.default_rng(random_state)
    v_w = rng.standard_normal(d).astype(FLOATX)
    v_b = FLOATX(rng.standard_normal())
    norm = math.sqrt(float(np.dot(v_w, v_w)) + float(v_b * v_b))
    v_w /= FLOATX(max(norm, EPS))
    v_b /= FLOATX(max(norm, EPS))

    eig = 1.0
    for _ in range(max(2, n_iter)):
        u = X @ v_w + v_b
        z_w = X.T @ u
        z_b = FLOATX(np.sum(u, dtype=np.float64))
        z_norm = math.sqrt(float(np.dot(z_w, z_w)) + float(z_b * z_b))
        if z_norm <= EPS:
            return 1.0
        v_w = (z_w / FLOATX(z_norm)).astype(FLOATX, copy=False)
        v_b = FLOATX(z_b / z_norm)
        eig = float(np.dot(u, u))
    return max(1e-8, 1.05 * eig / float(n))


def optimize_intercept_only(
    Y: np.ndarray, n_iter: int = 80, tol: float = 1e-7
) -> Tuple[np.ndarray, np.ndarray]:
    n, q = Y.shape
    b = np.zeros(q, dtype=FLOATX)
    mask = Y > 0
    residual = np.zeros((n, q), dtype=FLOATX)
    for _ in range(n_iter):
        P = np.broadcast_to(b, (n, q))
        _, residual = ambiguity_loss_from_predictions(P, mask)
        grad = residual.mean(axis=0, dtype=np.float64).astype(FLOATX)
        b_new = b - grad
        if np.linalg.norm(b_new - b) <= tol * max(1.0, np.linalg.norm(b)):
            b = b_new
            break
        b = b_new
    P = np.broadcast_to(b, (n, q))
    _, residual = ambiguity_loss_from_predictions(P, mask)
    return b.astype(FLOATX, copy=False), residual


def estimate_lambda_max(X: np.ndarray, Y: np.ndarray, intercept_iterations: int) -> Tuple[float, np.ndarray]:
    b0, residual = optimize_intercept_only(Y, n_iter=intercept_iterations)
    grad_w = (X.T @ residual) / FLOATX(X.shape[0])
    lambda_max = float(np.max(np.linalg.norm(grad_w, axis=1)))
    return max(lambda_max, 1e-8), b0


def fit_intercept_only_reference(
    Y: np.ndarray,
    gamma: float,
    base_fit: FitResult,
    config: SelectorConfig,
    initial_b: Optional[np.ndarray] = None,
) -> InterceptOnlyFit:
    """Fit F0*=min_b F(0,b) and attach a strong-convexity gap bound.

    This specialized problem costs O(nq) per iteration and never touches X.
    The group penalty is zero because W is fixed to zero.
    """
    start = time.perf_counter()
    if initial_b is None:
        b, _ = optimize_intercept_only(
            Y,
            n_iter=config.intercept_iterations,
        )
    else:
        b = np.asarray(initial_b, dtype=FLOATX).copy()
    mask = Y > 0
    step = 0.98 / (1.0 + gamma)
    objective = float("inf")
    gap_upper = float("inf")
    converged = False
    performed = 0

    for it in range(max(1, config.intercept_reference_max_iter)):
        P = np.broadcast_to(b, Y.shape)
        loss, residual = ambiguity_loss_from_predictions(P, mask)
        grad = residual.mean(axis=0, dtype=np.float64).astype(FLOATX)
        grad += FLOATX(gamma) * b
        objective = loss + 0.5 * gamma * float(np.dot(b, b))
        gap_upper = max(
            config.optimization_slack_floor,
            float(np.dot(grad, grad)) / (2.0 * gamma),
        )
        feature_gain_estimate = max(0.0, objective - base_fit.objective)
        epsilon_scale = max(
            config.epsilon_ratio * feature_gain_estimate,
            config.epsilon_ratio * max(abs(base_fit.objective), EPS) * 1e-6,
            EPS,
        )
        performed = it + 1
        if (
            performed >= config.min_iter
            and gap_upper / epsilon_scale
            <= config.certificate_gap_ratio_target
        ):
            converged = True
            break
        b = (b - FLOATX(step) * grad).astype(FLOATX, copy=False)

    # Report the objective/certificate at the returned iterate.
    P = np.broadcast_to(b, Y.shape)
    loss, residual = ambiguity_loss_from_predictions(P, mask)
    grad = residual.mean(axis=0, dtype=np.float64).astype(FLOATX)
    grad += FLOATX(gamma) * b
    objective = loss + 0.5 * gamma * float(np.dot(b, b))
    gap_upper = max(
        config.optimization_slack_floor,
        float(np.dot(grad, grad)) / (2.0 * gamma),
    )
    feature_gain_estimate = max(0.0, objective - base_fit.objective)
    epsilon_scale = max(
        config.epsilon_ratio * feature_gain_estimate,
        config.epsilon_ratio * max(abs(base_fit.objective), EPS) * 1e-6,
        EPS,
    )
    converged = bool(
        gap_upper / epsilon_scale <= config.certificate_gap_ratio_target
    )
    return InterceptOnlyFit(
        b=b,
        objective=float(objective),
        optimization_gap_upper=float(gap_upper),
        n_iter=performed,
        elapsed_seconds=time.perf_counter() - start,
        certificate_converged=converged,
    )


def _normalize_fixed_zero_rows(
    fixed_zero: Optional[int | Sequence[int] | np.ndarray],
    d: int,
) -> Optional[np.ndarray]:
    if fixed_zero is None:
        return None
    rows = np.unique(np.atleast_1d(np.asarray(fixed_zero, dtype=np.int64)))
    if rows.size == 0:
        return None
    if rows[0] < 0 or rows[-1] >= d:
        raise ValueError(f"fixed_zero contains an index outside [0, {d}).")
    return rows


class ConvexPartialLabelSelector:
    def __init__(self, lam: float, gamma: float, config: SelectorConfig, lipschitz: float):
        if gamma <= 0:
            raise ValueError("gamma must be positive for strong-convexity certificates.")
        self.lam = float(lam)
        self.gamma = float(gamma)
        self.config = config
        self.lipschitz = float(lipschitz)

    def _smooth_objective_and_grad(
        self, X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        P = X @ W
        P += b
        loss, residual = ambiguity_loss_from_predictions(P, Y > 0)
        ridge = 0.5 * self.gamma * (
            float(np.sum(W * W, dtype=np.float64)) + float(np.dot(b, b))
        )
        grad_w = (X.T @ residual) / FLOATX(X.shape[0])
        grad_w += FLOATX(self.gamma) * W
        grad_b = residual.mean(axis=0, dtype=np.float64).astype(FLOATX)
        grad_b += FLOATX(self.gamma) * b
        return loss + ridge, grad_w.astype(FLOATX, copy=False), grad_b, residual

    def _objective(self, X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        P = X @ W
        P += b
        loss, _ = ambiguity_loss_from_predictions(P, Y > 0)
        ridge = 0.5 * self.gamma * (
            float(np.sum(W * W, dtype=np.float64)) + float(np.dot(b, b))
        )
        smooth = loss + ridge
        penalty = self.lam * float(np.sum(np.linalg.norm(W, axis=1), dtype=np.float64))
        return smooth + penalty, smooth

    def _prox_rows(
        self,
        V: np.ndarray,
        step: float,
        fixed_zero: Optional[np.ndarray],
    ) -> np.ndarray:
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        scale = np.maximum(0.0, 1.0 - step * self.lam / (norms + EPS))
        out = (V * scale).astype(FLOATX, copy=False)
        if fixed_zero is not None:
            out[fixed_zero, :] = 0.0
        return out

    def _stationarity_residual(
        self,
        grad_w: np.ndarray,
        grad_b: np.ndarray,
        W: np.ndarray,
        fixed_zero: Optional[np.ndarray],
    ) -> float:
        row_norm = np.linalg.norm(W, axis=1)
        residual_w = np.empty_like(grad_w)
        active = row_norm > 1e-10
        if np.any(active):
            residual_w[active] = grad_w[active] + (
                self.lam * W[active] / row_norm[active, None]
            )
        inactive = ~active
        if np.any(inactive):
            g = grad_w[inactive]
            gnorm = np.linalg.norm(g, axis=1)
            factor = np.maximum(0.0, 1.0 - self.lam / (gnorm + EPS))
            residual_w[inactive] = g * factor[:, None]
        if fixed_zero is not None:
            # The fixed row is not a free direction in the constrained problem.
            residual_w[fixed_zero, :] = 0.0
        return math.sqrt(
            float(np.sum(residual_w * residual_w, dtype=np.float64))
            + float(np.dot(grad_b, grad_b))
        )

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        initial_W: Optional[np.ndarray] = None,
        initial_b: Optional[np.ndarray] = None,
        fixed_zero: Optional[int | Sequence[int] | np.ndarray] = None,
        max_iter: Optional[int] = None,
        certify: bool = True,
    ) -> FitResult:
        start = time.perf_counter()
        n, d = X.shape
        q = Y.shape[1]
        iterations = int(max_iter if max_iter is not None else self.config.max_iter)
        fixed_zero = _normalize_fixed_zero_rows(fixed_zero, d)

        if initial_W is None:
            W = np.zeros((d, q), dtype=FLOATX)
        else:
            W = np.asarray(initial_W, dtype=FLOATX).copy()
        if initial_b is None:
            b = np.zeros(q, dtype=FLOATX)
        else:
            b = np.asarray(initial_b, dtype=FLOATX).copy()
        if fixed_zero is not None:
            W[fixed_zero, :] = 0.0

        Z_w = W.copy()
        Z_b = b.copy()
        momentum = 1.0
        step = 0.98 / (self.lipschitz + self.gamma)
        performed = 0

        for it in range(iterations):
            _, grad_w, grad_b, _ = self._smooth_objective_and_grad(X, Y, Z_w, Z_b)
            W_new = self._prox_rows(Z_w - FLOATX(step) * grad_w, step, fixed_zero)
            b_new = (Z_b - FLOATX(step) * grad_b).astype(FLOATX, copy=False)

            delta_norm = math.sqrt(
                float(np.sum((W_new - W) ** 2, dtype=np.float64))
                + float(np.dot(b_new - b, b_new - b))
            )
            base_norm = math.sqrt(
                float(np.sum(W * W, dtype=np.float64)) + float(np.dot(b, b))
            )

            # Adaptive FISTA restart suppresses oscillations with little overhead.
            restart_inner = float(np.sum((W_new - W) * (Z_w - W_new), dtype=np.float64))
            restart_inner += float(np.dot(b_new - b, Z_b - b_new))
            if restart_inner > 0.0:
                momentum_new = 1.0
                Z_w = W_new.copy()
                Z_b = b_new.copy()
            else:
                momentum_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
                factor = FLOATX((momentum - 1.0) / momentum_new)
                Z_w = (W_new + factor * (W_new - W)).astype(FLOATX, copy=False)
                Z_b = (b_new + factor * (b_new - b)).astype(FLOATX, copy=False)
                if fixed_zero is not None:
                    Z_w[fixed_zero, :] = 0.0

            W, b = W_new, b_new
            momentum = momentum_new
            performed = it + 1
            if performed >= self.config.min_iter and delta_norm <= self.config.tol * max(1.0, base_norm):
                break

        # A few plain proximal-gradient refinement steps improve the numerical
        # certificate and are used only for final/off fits, not grid-search fits.
        if certify:
            for _ in range(self.config.certificate_refine_iterations):
                _, grad_w, grad_b, _ = self._smooth_objective_and_grad(X, Y, W, b)
                W_new = self._prox_rows(W - FLOATX(step) * grad_w, step, fixed_zero)
                b_new = (b - FLOATX(step) * grad_b).astype(FLOATX, copy=False)
                update = math.sqrt(
                    float(np.sum((W_new - W) ** 2, dtype=np.float64))
                    + float(np.dot(b_new - b, b_new - b))
                )
                W, b = W_new, b_new
                performed += 1
                if update <= 0.25 * self.config.tol * max(1.0, np.linalg.norm(W)):
                    break

        objective, smooth = self._objective(X, Y, W, b)
        _, grad_w, grad_b, _ = self._smooth_objective_and_grad(X, Y, W, b)
        residual_norm = self._stationarity_residual(grad_w, grad_b, W, fixed_zero)
        opt_gap = max(
            self.config.optimization_slack_floor,
            residual_norm * residual_norm / (2.0 * self.gamma),
        )
        return FitResult(
            W=W.astype(FLOATX, copy=False),
            b=b.astype(FLOATX, copy=False),
            objective=float(objective),
            smooth_objective=float(smooth),
            stationarity_residual=float(residual_norm),
            optimization_gap_upper=float(opt_gap),
            n_iter=performed,
            elapsed_seconds=time.perf_counter() - start,
        )


def fit_with_adaptive_certificate(
    selector: ConvexPartialLabelSelector,
    X: np.ndarray,
    Y: np.ndarray,
    initial_W: Optional[np.ndarray] = None,
    initial_b: Optional[np.ndarray] = None,
    fixed_zero: Optional[int | Sequence[int] | np.ndarray] = None,
    initial_max_iter: Optional[int] = None,
    epsilon_reference: Optional[float] = None,
) -> FitResult:
    """Continue a convex fit until its numerical gap is small versus epsilon.

    Before the feature-gain reference is available, the unconstrained fit uses
    epsilon_ratio * objective as a numerical scale.  Callers then perform one
    feature-relative continuation when needed.  Constrained fits receive the
    fixed epsilon scale explicitly.
    """
    config = selector.config
    fixed_zero = _normalize_fixed_zero_rows(fixed_zero, X.shape[1])
    current = selector.fit(
        X,
        Y,
        initial_W=initial_W,
        initial_b=initial_b,
        fixed_zero=fixed_zero,
        max_iter=initial_max_iter,
        certify=True,
    )
    total_iter = current.n_iter
    total_seconds = current.elapsed_seconds
    rounds_used = 0

    def _epsilon(fit: FitResult) -> float:
        if epsilon_reference is not None:
            return max(float(epsilon_reference), EPS)
        return max(config.epsilon_ratio * max(abs(fit.objective), EPS), EPS)

    epsilon = _epsilon(current)
    ratio = current.optimization_gap_upper / epsilon

    for round_id in range(config.adaptive_refinement_rounds):
        if ratio <= config.certificate_gap_ratio_target:
            break
        tighter_tol = max(
            config.adaptive_min_tol,
            config.tol * config.adaptive_tolerance_shrink ** (round_id + 1),
        )
        tighter_config = replace(
            config,
            tol=tighter_tol,
            certificate_refine_iterations=max(
                config.certificate_refine_iterations,
                config.adaptive_refinement_max_iter // 4,
            ),
        )
        refinement_selector = ConvexPartialLabelSelector(
            selector.lam,
            selector.gamma,
            tighter_config,
            selector.lipschitz,
        )
        refined = refinement_selector.fit(
            X,
            Y,
            initial_W=current.W,
            initial_b=current.b,
            fixed_zero=fixed_zero,
            max_iter=config.adaptive_refinement_max_iter,
            certify=True,
        )
        total_iter += refined.n_iter
        total_seconds += refined.elapsed_seconds
        current = refined
        rounds_used = round_id + 1
        epsilon = _epsilon(current)
        ratio = current.optimization_gap_upper / epsilon
        if config.verbose:
            if fixed_zero is None:
                fit_name = "baseline"
            elif fixed_zero.size == 1:
                fit_name = f"off[{int(fixed_zero[0])}]"
            else:
                fit_name = f"group-off[size={fixed_zero.size}]"
            print(
                f"[Certificate] {fit_name}, refinement={rounds_used}, "
                f"gap/epsilon={ratio:.3e}, target="
                f"{config.certificate_gap_ratio_target:.3e}, tol={tighter_tol:.1e}"
            )

    return replace(
        current,
        n_iter=total_iter,
        elapsed_seconds=total_seconds,
        optimization_gap_ratio=float(ratio),
        certificate_converged=bool(ratio <= config.certificate_gap_ratio_target),
        adaptive_refinement_rounds=rounds_used,
    )


def build_epsilon_reference(
    base_fit: FitResult,
    intercept_fit: InterceptOnlyFit,
    config: SelectorConfig,
) -> EpsilonReference:
    """Build safe epsilon bounds from certified objective intervals."""
    mode = config.epsilon_mode.lower().strip()
    if mode not in {"feature_gain", "total_objective"}:
        raise ValueError(
            "epsilon_mode must be one of: feature_gain, total_objective"
        )

    if mode == "total_objective":
        epsilon = config.epsilon_ratio * max(abs(base_fit.objective), EPS)
        return EpsilonReference(
            mode=mode,
            epsilon=float(epsilon),
            epsilon_lower=float(epsilon),
            epsilon_upper=float(epsilon),
            feature_gain=float("nan"),
            feature_gain_lower=float("nan"),
            feature_gain_upper=float("nan"),
            intercept_objective=float(intercept_fit.objective),
            intercept_optimization_gap_upper=float(
                intercept_fit.optimization_gap_upper
            ),
            intercept_certificate_converged=bool(
                intercept_fit.certificate_converged
            ),
            intercept_n_iter=int(intercept_fit.n_iter),
            intercept_elapsed_seconds=float(intercept_fit.elapsed_seconds),
        )

    # F* is in [Fhat-g, Fhat] and F0* is in [F0hat-g0, F0hat].
    # Subtracting these intervals gives a guaranteed interval for the total
    # feature-attributable utility B=F0*-F*.
    feature_gain = max(0.0, intercept_fit.objective - base_fit.objective)
    feature_gain_lower = max(
        0.0,
        intercept_fit.objective
        - intercept_fit.optimization_gap_upper
        - base_fit.objective,
    )
    feature_gain_upper = max(
        feature_gain_lower,
        intercept_fit.objective
        - base_fit.objective
        + base_fit.optimization_gap_upper,
    )
    return EpsilonReference(
        mode=mode,
        epsilon=float(config.epsilon_ratio * feature_gain),
        epsilon_lower=float(config.epsilon_ratio * feature_gain_lower),
        epsilon_upper=float(config.epsilon_ratio * feature_gain_upper),
        feature_gain=float(feature_gain),
        feature_gain_lower=float(feature_gain_lower),
        feature_gain_upper=float(feature_gain_upper),
        intercept_objective=float(intercept_fit.objective),
        intercept_optimization_gap_upper=float(
            intercept_fit.optimization_gap_upper
        ),
        intercept_certificate_converged=bool(
            intercept_fit.certificate_converged
        ),
        intercept_n_iter=int(intercept_fit.n_iter),
        intercept_elapsed_seconds=float(intercept_fit.elapsed_seconds),
    )


# -----------------------------------------------------------------------------
# Necessary-feature certification
# -----------------------------------------------------------------------------


def certify_necessary_features(
    X: np.ndarray,
    Y: np.ndarray,
    selector: ConvexPartialLabelSelector,
    base_fit: FitResult,
    config: SelectorConfig,
    epsilon_reference: EpsilonReference,
) -> NecessityResult:
    """Classify features with safe lower/upper deletion-gap bounds.

    The lower bound follows from gamma-strong convexity.  The upper bound is
    obtained from a feasible zero-row model and the 1-smoothness of squared
    distance to a closed convex set.  Exact mode reoptimizes only rows whose
    bounds straddle epsilon.
    """
    start = time.perf_counter()
    W, b = base_fit.W, base_fit.b
    n, d = X.shape
    q = Y.shape[1]
    epsilon = epsilon_reference.epsilon
    epsilon_lower = epsilon_reference.epsilon_lower
    epsilon_upper = epsilon_reference.epsilon_upper

    row_norm = np.linalg.norm(W, axis=1).astype(np.float64)
    opt_radius = math.sqrt(2.0 * base_fit.optimization_gap_upper / selector.gamma)
    off_gap_lower = 0.5 * selector.gamma * np.maximum(0.0, row_norm - opt_radius) ** 2

    # A cheap smoothness upper bound for zeroing each row at the approximate
    # optimum.  It requires one final gradient, not d matrix multiplications.
    _, grad_w, _, _ = selector._smooth_objective_and_grad(X, Y, W, b)
    grad_loss = grad_w - FLOATX(selector.gamma) * W
    feature_curvature = np.sum(X * X, axis=0, dtype=np.float64) / float(n)
    linear_term = -np.sum(grad_loss * W, axis=1, dtype=np.float64)
    zero_change_upper = (
        linear_term
        + 0.5 * feature_curvature * row_norm**2
        - selector.lam * row_norm
        - 0.5 * selector.gamma * row_norm**2
    )
    off_gap_upper = np.maximum(0.0, zero_change_upper + base_fit.optimization_gap_upper)
    off_gap_estimate = np.full(d, np.nan, dtype=np.float64)

    # Tighten the upper bound by evaluating the actually zeroed feasible model
    # for a limited number of ambiguous rows.  This uses rank-one prediction
    # updates and avoids recomputing X @ W.
    initial_ambiguous = np.flatnonzero(
        (off_gap_lower <= epsilon_upper) & (off_gap_upper > epsilon_lower)
    )
    if initial_ambiguous.size:
        priority = initial_ambiguous[
            np.argsort(-off_gap_lower[initial_ambiguous], kind="stable")
        ]
        zero_eval_idx = priority[: max(0, config.max_zero_bound_evaluations)]
        if zero_eval_idx.size:
            P_base = X @ W
            P_base += b
            ridge_base = 0.5 * selector.gamma * (
                float(np.sum(W * W, dtype=np.float64)) + float(np.dot(b, b))
            )
            penalty_base = selector.lam * float(
                np.sum(np.linalg.norm(W, axis=1), dtype=np.float64)
            )
            for k in zero_eval_idx:
                if row_norm[k] <= 1e-14:
                    feasible_obj = base_fit.objective
                else:
                    P_zero = P_base - X[:, k : k + 1] * W[k : k + 1, :]
                    loss_zero, _ = ambiguity_loss_from_predictions(P_zero, Y > 0)
                    ridge_zero = ridge_base - 0.5 * selector.gamma * row_norm[k] ** 2
                    penalty_zero = penalty_base - selector.lam * row_norm[k]
                    feasible_obj = loss_zero + ridge_zero + penalty_zero
                exact_feasible_ub = max(
                    0.0,
                    feasible_obj - base_fit.objective + base_fit.optimization_gap_upper,
                )
                off_gap_upper[k] = min(off_gap_upper[k], exact_feasible_ub)

    # Interval-safe three-way decision.  Necessary uses the largest admissible
    # tolerance; nonnecessary uses the smallest admissible tolerance.
    necessary = off_gap_lower > epsilon_upper
    nonnecessary = off_gap_upper <= epsilon_lower
    unresolved = ~(necessary | nonnecessary)

    mode = config.necessity_mode.lower().strip()
    if mode not in {"auto", "fast", "exact"}:
        raise ValueError("necessity_mode must be one of: auto, fast, exact")

    candidates = np.flatnonzero(unresolved)
    if mode == "fast":
        exact_budget = 0
        mode_used = "fast"
    elif mode == "exact":
        exact_budget = len(candidates)
        mode_used = "exact"
    else:
        ndq = float(n) * float(d) * float(q)
        if d <= 100:
            exact_budget = len(candidates)
            mode_used = "auto-exact"
        elif ndq <= config.auto_skip_exact_ndq:
            exact_budget = min(len(candidates), config.auto_max_exact_refits)
            mode_used = "auto-hybrid"
        else:
            exact_budget = 0
            mode_used = "auto-fast-highdim"

    exact_refits = 0
    exact_refits_converged = 0
    max_off_gap_ratio = 0.0
    if exact_budget > 0 and len(candidates) > 0:
        # Features closest to being certified by the lower bound are attempted
        # first, with the upper bound used as a stable tie-breaker.
        order = np.lexsort((-off_gap_upper[candidates], -off_gap_lower[candidates]))
        exact_candidates = candidates[order[:exact_budget]]
        for pos, k in enumerate(exact_candidates, start=1):
            initial_w = W.copy()
            initial_w[k, :] = 0.0
            off_fit = fit_with_adaptive_certificate(
                selector,
                X,
                Y,
                initial_W=initial_w,
                initial_b=b,
                fixed_zero=int(k),
                initial_max_iter=config.off_max_iter,
                epsilon_reference=max(epsilon_lower, epsilon, EPS),
            )
            gap_lower = (
                off_fit.objective
                - off_fit.optimization_gap_upper
                - base_fit.objective
            )
            gap_upper = (
                off_fit.objective
                - base_fit.objective
                + base_fit.optimization_gap_upper
            )
            off_gap_lower[k] = max(0.0, gap_lower)
            off_gap_upper[k] = max(off_gap_lower[k], gap_upper)
            off_gap_estimate[k] = off_fit.objective - base_fit.objective
            necessary[k] = off_gap_lower[k] > epsilon_upper
            nonnecessary[k] = off_gap_upper[k] <= epsilon_lower
            unresolved[k] = not (necessary[k] or nonnecessary[k])
            exact_refits += 1
            exact_refits_converged += int(off_fit.certificate_converged)
            max_off_gap_ratio = max(
                max_off_gap_ratio,
                float(off_fit.optimization_gap_ratio),
            )
            if config.verbose:
                print(
                    f"[Necessity] exact {pos}/{len(exact_candidates)} feature={k}, "
                    f"gap=[{off_gap_lower[k]:.3e}, {off_gap_upper[k]:.3e}], "
                    f"epsilon=[{epsilon_lower:.3e}, {epsilon_upper:.3e}]"
                )

    return NecessityResult(
        necessary=necessary,
        unresolved=unresolved,
        off_gap_lower=off_gap_lower,
        off_gap_upper=off_gap_upper,
        off_gap_estimate=off_gap_estimate,
        epsilon=float(epsilon),
        epsilon_lower=float(epsilon_lower),
        epsilon_upper=float(epsilon_upper),
        epsilon_mode=epsilon_reference.mode,
        feature_gain=epsilon_reference.feature_gain,
        feature_gain_lower=epsilon_reference.feature_gain_lower,
        feature_gain_upper=epsilon_reference.feature_gain_upper,
        intercept_objective=epsilon_reference.intercept_objective,
        intercept_optimization_gap_upper=(
            epsilon_reference.intercept_optimization_gap_upper
        ),
        intercept_certificate_converged=(
            epsilon_reference.intercept_certificate_converged
        ),
        intercept_n_iter=epsilon_reference.intercept_n_iter,
        intercept_elapsed_seconds=(
            epsilon_reference.intercept_elapsed_seconds
        ),
        exact_refits=exact_refits,
        exact_refits_converged=exact_refits_converged,
        max_off_optimization_gap_ratio=float(max_off_gap_ratio),
        mode_used=mode_used,
        elapsed_seconds=time.perf_counter() - start,
    )


def _group_diagnostic_sizes(d: int, config: SelectorConfig) -> List[int]:
    fraction_limit = max(
        1,
        int(math.ceil(d * max(0.0, config.group_diagnostic_max_fraction))),
    )
    limit = min(d, max(1, config.group_diagnostic_max_features), fraction_limit)
    sizes: List[int] = []
    size = 1
    while size <= limit:
        sizes.append(size)
        size *= 2
    if sizes[-1] != limit:
        sizes.append(limit)
    return sizes


def certify_group_deletion(
    X: np.ndarray,
    Y: np.ndarray,
    selector: ConvexPartialLabelSelector,
    base_fit: FitResult,
    singleton_result: NecessityResult,
    group_indices: Sequence[int] | np.ndarray,
    precomputed_predictions: Optional[np.ndarray] = None,
    precomputed_ridge: Optional[float] = None,
    precomputed_penalty: Optional[float] = None,
) -> GroupDiagnosticResult:
    """Certify whether all rows in one feature group can be removed together."""
    start = time.perf_counter()
    group = _normalize_fixed_zero_rows(group_indices, X.shape[1])
    if group is None:
        raise ValueError("group_indices must contain at least one feature.")

    W, b = base_fit.W, base_fit.b
    epsilon = singleton_result.epsilon
    epsilon_lower = singleton_result.epsilon_lower
    epsilon_upper = singleton_result.epsilon_upper
    group_weights = W[group, :]
    group_weight_norm = float(np.linalg.norm(group_weights))
    opt_radius = math.sqrt(
        2.0 * base_fit.optimization_gap_upper / selector.gamma
    )
    gap_lower = 0.5 * selector.gamma * max(
        0.0,
        group_weight_norm - opt_radius,
    ) ** 2

    # Feasible group-zero model: update predictions only by the selected block
    # instead of recomputing X @ W from scratch.
    if precomputed_predictions is None:
        P_base = X @ W
        P_base += b
    else:
        P_base = precomputed_predictions
    P_zero = P_base - X[:, group] @ group_weights
    loss_zero, _ = ambiguity_loss_from_predictions(P_zero, Y > 0)
    group_sq_norm = float(np.sum(group_weights * group_weights, dtype=np.float64))
    group_row_norm_sum = float(
        np.sum(np.linalg.norm(group_weights, axis=1), dtype=np.float64)
    )
    ridge_base = (
        0.5
        * selector.gamma
        * (float(np.sum(W * W, dtype=np.float64)) + float(np.dot(b, b)))
        if precomputed_ridge is None
        else float(precomputed_ridge)
    )
    penalty_base = (
        selector.lam
        * float(np.sum(np.linalg.norm(W, axis=1), dtype=np.float64))
        if precomputed_penalty is None
        else float(precomputed_penalty)
    )
    feasible_objective = (
        loss_zero
        + ridge_base
        - 0.5 * selector.gamma * group_sq_norm
        + penalty_base
        - selector.lam * group_row_norm_sum
    )
    gap_upper = max(
        gap_lower,
        feasible_objective
        - base_fit.objective
        + base_fit.optimization_gap_upper,
    )

    exact_refit = False
    gap_estimate = float("nan")
    certificate_converged = bool(base_fit.certificate_converged)
    optimization_gap_ratio = float(base_fit.optimization_gap_ratio)
    if gap_lower <= epsilon_upper and gap_upper > epsilon_lower:
        initial_w = W.copy()
        initial_w[group, :] = 0.0
        off_fit = fit_with_adaptive_certificate(
            selector,
            X,
            Y,
            initial_W=initial_w,
            initial_b=b,
            fixed_zero=group,
            initial_max_iter=selector.config.off_max_iter,
            epsilon_reference=max(epsilon_lower, epsilon, EPS),
        )
        gap_lower = max(
            0.0,
            off_fit.objective
            - off_fit.optimization_gap_upper
            - base_fit.objective,
        )
        gap_upper = max(
            gap_lower,
            off_fit.objective
            - base_fit.objective
            + base_fit.optimization_gap_upper,
        )
        gap_estimate = float(off_fit.objective - base_fit.objective)
        exact_refit = True
        certificate_converged = bool(off_fit.certificate_converged)
        optimization_gap_ratio = float(off_fit.optimization_gap_ratio)

    if gap_lower > epsilon_upper:
        status = "JointlyNecessary"
    elif gap_upper <= epsilon_lower:
        status = "NonNecessary"
    else:
        status = "Unresolved"

    singleton_lower_sum = float(
        np.sum(singleton_result.off_gap_lower[group], dtype=np.float64)
    )
    singleton_upper_sum = float(
        np.sum(singleton_result.off_gap_upper[group], dtype=np.float64)
    )
    # This value is a safe lower bound on interaction beyond the sum of all
    # singleton deletion costs.  Positive values certify genuine group synergy.
    certified_synergy_lower = float(gap_lower - singleton_upper_sum)

    return GroupDiagnosticResult(
        group_size=int(group.size),
        feature_indices=group,
        group_weight_norm=group_weight_norm,
        gap_lower=float(gap_lower),
        gap_upper=float(gap_upper),
        gap_estimate=gap_estimate,
        epsilon=float(epsilon),
        epsilon_lower=float(epsilon_lower),
        epsilon_upper=float(epsilon_upper),
        status=status,
        exact_refit=exact_refit,
        certificate_converged=certificate_converged,
        optimization_gap_ratio=optimization_gap_ratio,
        sum_singleton_lower=singleton_lower_sum,
        sum_singleton_upper=singleton_upper_sum,
        certified_synergy_lower=certified_synergy_lower,
        elapsed_seconds=time.perf_counter() - start,
    )


def cumulative_group_diagnostic(
    X: np.ndarray,
    Y: np.ndarray,
    selector: ConvexPartialLabelSelector,
    base_fit: FitResult,
    singleton_result: NecessityResult,
    ranking: np.ndarray,
) -> pd.DataFrame:
    """Find the first certified nested top-ranked group.

    Powers of two locate a crossing cheaply.  Once a nonnecessary/necessary
    bracket is found, binary refinement identifies the smallest certified
    prefix in that bracket.
    """
    rows: List[Dict[str, object]] = []
    sizes = _group_diagnostic_sizes(X.shape[1], selector.config)
    P_base = X @ base_fit.W
    P_base += base_fit.b
    ridge_base = 0.5 * selector.gamma * (
        float(np.sum(base_fit.W * base_fit.W, dtype=np.float64))
        + float(np.dot(base_fit.b, base_fit.b))
    )
    penalty_base = selector.lam * float(
        np.sum(np.linalg.norm(base_fit.W, axis=1), dtype=np.float64)
    )
    cache: Dict[int, GroupDiagnosticResult] = {}

    def evaluate(size: int) -> GroupDiagnosticResult:
        if size not in cache:
            group = np.asarray(ranking[:size], dtype=np.int64)
            cache[size] = certify_group_deletion(
                X,
                Y,
                selector,
                base_fit,
                singleton_result,
                group,
                precomputed_predictions=P_base,
                precomputed_ridge=ridge_base,
                precomputed_penalty=penalty_base,
            )
        return cache[size]

    evaluated: List[Tuple[int, str]] = []
    first_necessary: Optional[int] = None
    previous_size = 0
    previous_status = "NonNecessary"
    for size in sizes:
        result = evaluate(size)
        evaluated.append((size, "Exponential"))
        if result.status == "JointlyNecessary":
            first_necessary = size
            if (
                selector.config.group_diagnostic_binary_refine
                and previous_status == "NonNecessary"
                and size - previous_size > 1
            ):
                low, high = previous_size, size
                while high - low > 1:
                    middle = (low + high) // 2
                    middle_result = evaluate(middle)
                    evaluated.append((middle, "BinaryRefine"))
                    if middle_result.status == "JointlyNecessary":
                        high = middle
                    elif middle_result.status == "NonNecessary":
                        low = middle
                    else:
                        # An unresolved point cannot prove that all smaller
                        # prefixes are nonnecessary, so keep the conservative
                        # certified necessary endpoint.
                        low = middle
                first_necessary = high
            if selector.config.group_diagnostic_stop_after_first_necessary:
                break
        previous_size = size
        previous_status = result.status

    # Keep the actual evaluation order for runtime auditing.  The explicit
    # group size column makes the non-monotone binary-search order unambiguous.
    for group_index, (size, search_phase) in enumerate(evaluated, start=1):
        result = cache[size]
        group = np.asarray(ranking[:size], dtype=np.int64)
        rows.append(
            {
                "GroupIndex": group_index,
                "SearchPhase": search_phase,
                "GroupSize": result.group_size,
                "GroupFraction(%)": 100.0 * result.group_size / X.shape[1],
                "RankStart": 1,
                "RankEnd": result.group_size,
                "FeatureIndices0": ";".join(map(str, result.feature_indices)),
                "FeatureIndices1": ";".join(
                    map(str, result.feature_indices + 1)
                ),
                "GroupWeightNorm": result.group_weight_norm,
                "GapLowerBound": result.gap_lower,
                "GapUpperBound": result.gap_upper,
                "GapEstimate": result.gap_estimate,
                "Epsilon": result.epsilon,
                "EpsilonLower": result.epsilon_lower,
                "EpsilonUpper": result.epsilon_upper,
                "GapLower/Epsilon": result.gap_lower / max(result.epsilon, EPS),
                "GapUpper/Epsilon": result.gap_upper / max(result.epsilon, EPS),
                "GapLower/EpsilonUpper": (
                    result.gap_lower / max(result.epsilon_upper, EPS)
                ),
                "GapUpper/EpsilonLower": (
                    result.gap_upper / max(result.epsilon_lower, EPS)
                ),
                "RhoLower": (
                    result.gap_lower
                    / max(singleton_result.feature_gain_upper, EPS)
                ),
                "RhoUpper": (
                    result.gap_upper
                    / max(singleton_result.feature_gain_lower, EPS)
                ),
                "RhoEstimate": (
                    result.gap_estimate
                    / max(singleton_result.feature_gain, EPS)
                ),
                "Status": result.status,
                "FirstCertifiedNecessaryPrefix": bool(
                    first_necessary is not None and size == first_necessary
                ),
                "ExactRefit": result.exact_refit,
                "CertificateConverged": result.certificate_converged,
                "OptimizationGapRatio": result.optimization_gap_ratio,
                "SumSingletonLower": result.sum_singleton_lower,
                "SumSingletonUpper": result.sum_singleton_upper,
                "CertifiedSynergyLower": result.certified_synergy_lower,
                "CertifiedSynergy": result.certified_synergy_lower > 0.0,
                "ElapsedSeconds": result.elapsed_seconds,
            }
        )
        if selector.config.verbose:
            print(
                f"[GroupDiagnostic] size={result.group_size}, "
                f"gap/epsilon=["
                f"{result.gap_lower / max(result.epsilon, EPS):.3e}, "
                f"{result.gap_upper / max(result.epsilon, EPS):.3e}], "
                f"status={result.status}, exact={result.exact_refit}"
            )
    return pd.DataFrame(rows)


def necessity_first_ranking(W: np.ndarray, necessary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    importance = np.linalg.norm(W, axis=1).astype(np.float64)
    # np.lexsort uses the last key as primary: necessary first, then norm.
    ranking = np.lexsort((-importance, -necessary.astype(np.int8)))
    return ranking.astype(np.int64), importance


def feature_detail_frame(
    fit: FitResult,
    necessity: NecessityResult,
) -> pd.DataFrame:
    ranking, importance = necessity_first_ranking(fit.W, necessity.necessary)
    rank_position = np.empty(len(ranking), dtype=np.int64)
    rank_position[ranking] = np.arange(1, len(ranking) + 1)
    return pd.DataFrame(
        {
            "FeatureIndex0": np.arange(fit.W.shape[0], dtype=np.int64),
            "FeatureIndex1": np.arange(1, fit.W.shape[0] + 1, dtype=np.int64),
            "Rank": rank_position,
            "IsNecessary": necessity.necessary.astype(np.int8),
            "IsUnresolved": necessity.unresolved.astype(np.int8),
            "WeightNorm": importance,
            "OffGapLowerBound": necessity.off_gap_lower,
            "OffGapUpperBound": necessity.off_gap_upper,
            "OffGapEstimate": necessity.off_gap_estimate,
            "Epsilon": necessity.epsilon,
            "EpsilonLower": necessity.epsilon_lower,
            "EpsilonUpper": necessity.epsilon_upper,
            "EpsilonMode": necessity.epsilon_mode,
            "FeatureGain": necessity.feature_gain,
            "FeatureGainLower": necessity.feature_gain_lower,
            "FeatureGainUpper": necessity.feature_gain_upper,
            "DeletionGap/FeatureGain": (
                necessity.off_gap_estimate
                / max(necessity.feature_gain, EPS)
            ),
            "RhoLower": (
                necessity.off_gap_lower
                / max(necessity.feature_gain_upper, EPS)
            ),
            "RhoUpper": (
                necessity.off_gap_upper
                / max(necessity.feature_gain_lower, EPS)
            ),
            "InterceptOnlyObjective": necessity.intercept_objective,
            "InterceptOnlyOptimizationGapUpper": (
                necessity.intercept_optimization_gap_upper
            ),
            "InterceptOnlyCertificateConverged": (
                necessity.intercept_certificate_converged
            ),
            "InterceptOnlyIterations": necessity.intercept_n_iter,
            "InterceptOnlySeconds": necessity.intercept_elapsed_seconds,
            "OptimizationGapRatio": fit.optimization_gap_ratio,
            "CertificateConverged": fit.certificate_converged,
            "AdaptiveRefinementRounds": fit.adaptive_refinement_rounds,
            "ExactRefitsConverged": necessity.exact_refits_converged,
            "MaxOffOptimizationGapRatio": necessity.max_off_optimization_gap_ratio,
        }
    ).sort_values("Rank", kind="stable")


# -----------------------------------------------------------------------------
# Downstream LinearSVC and the seven metrics from robust.py
# -----------------------------------------------------------------------------


class SafeOVRLinearSVC:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model_: Optional[OneVsRestClassifier] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "SafeOVRLinearSVC":
        dual_value = True if issparse(X) else "auto"
        estimator = LinearSVC(
            C=self.config.svm_c,
            tol=self.config.svm_tol,
            max_iter=self.config.svm_max_iter,
            dual=dual_value,
            random_state=self.config.random_state,
        )
        self.model_ = OneVsRestClassifier(estimator, n_jobs=self.config.svm_n_jobs)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Label .* is present in all training examples")
            warnings.filterwarnings("ignore", message="Label not .* is present in all training examples")
            # Do not suppress ConvergenceWarning: a final paper run must expose
            # any label-wise LinearSVC fit that still reaches max_iter.
            self.model_.fit(X, Y)
        return self

    def predict_scores_and_labels(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        scores = np.asarray(self.model_.decision_function(X), dtype=FLOATX)
        if scores.ndim == 1:
            scores = scores[:, None]
        labels = np.asarray(self.model_.predict(X), dtype=np.int32)
        if labels.ndim == 1:
            labels = labels[:, None]
        return scores, labels


def constant_no_feature_predictions(Y_train: np.ndarray, n_test: int) -> Tuple[np.ndarray, np.ndarray]:
    prevalence = Y_train.mean(axis=0, dtype=np.float64).astype(FLOATX)
    scores = np.broadcast_to(prevalence, (n_test, Y_train.shape[1])).copy()
    labels = (scores >= 0.5).astype(np.int32)
    return scores, labels


def one_error(y_true: np.ndarray, y_score: np.ndarray) -> float:
    top_idx = np.argmax(y_score, axis=1)
    hits = y_true[np.arange(y_true.shape[0]), top_idx]
    return float(np.mean(1.0 - hits))


def sample_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    valid = y_true.sum(axis=1) > 0
    if not np.any(valid):
        return 0.0
    y_true = y_true[valid]
    y_score = y_score[valid]
    values: List[float] = []
    for i in range(y_true.shape[0]):
        positives = np.flatnonzero(y_true[i] > 0)
        order = np.argsort(-y_score[i], kind="stable")
        hits = 0
        precision_sum = 0.0
        for rank, idx in enumerate(order, start=1):
            if y_true[i, idx] > 0:
                hits += 1
                precision_sum += hits / rank
        values.append(precision_sum / len(positives))
    return float(np.mean(values))


def normalized_coverage_error(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n, q = y_true.shape
    values: List[float] = []
    for i in range(n):
        positives = np.flatnonzero(y_true[i] > 0)
        if len(positives) == 0:
            continue
        order = np.argsort(-y_score[i], kind="stable")
        ranks = np.empty(q, dtype=np.int64)
        ranks[order] = np.arange(1, q + 1)
        coverage = int(np.max(ranks[positives]))
        denom = q - len(positives)
        values.append(0.0 if denom <= 0 else (coverage - len(positives)) / denom)
    return float(np.mean(values)) if values else 0.0


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "RankingLoss": float(label_ranking_loss(y_true, y_score)),
        "HammingLoss": float(hamming_loss(y_true, y_pred)),
        "CoverageError": float(normalized_coverage_error(y_true, y_score)),
        "OneError": float(one_error(y_true, y_score)),
        "AveragePrecision": float(sample_average_precision(y_true, y_score)),
        "MicroF1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "MacroF1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_feature_subset(
    X_train: np.ndarray,
    Y_train_partial: np.ndarray,
    X_test: np.ndarray,
    Y_test_target: np.ndarray,
    feature_indices: np.ndarray,
    eval_config: EvaluationConfig,
) -> Tuple[Dict[str, float], str]:
    feature_indices = np.asarray(feature_indices, dtype=np.int64)
    if feature_indices.size == 0:
        scores, labels = constant_no_feature_predictions(Y_train_partial, X_test.shape[0])
        classifier_mode = "ConstantNoFeature"
    else:
        X_train_selected = X_train[:, feature_indices]
        X_test_selected = X_test[:, feature_indices]
        density = float(np.count_nonzero(X_train_selected)) / float(
            max(1, X_train_selected.size)
        )
        if density <= eval_config.svm_sparse_density_threshold:
            X_train_selected = csr_matrix(X_train_selected)
            X_test_selected = csr_matrix(X_test_selected)
        classifier = SafeOVRLinearSVC(eval_config)
        classifier.fit(X_train_selected, Y_train_partial)
        scores, labels = classifier.predict_scores_and_labels(X_test_selected)
        classifier_mode = "LinearSVC"
    return evaluate_metrics(Y_test_target, labels, scores), classifier_mode


def choose_protocol_feature_count(d: int, percent: int) -> int:
    return max(1, int(math.ceil(d * percent / 100.0)))


def build_protocol_budgets(
    dataset_name: str,
    d: int,
    eval_config: EvaluationConfig,
) -> List[BudgetSpec]:
    """Return the paper-aligned feature budgets for one dataset.

    Water is the only stated exception in the previous manuscript: because it
    has 16 features, it is evaluated at feature counts 1,...,16.  All other
    datasets use the top 1%,...,20% rule, even when adjacent percentages round
    to the same feature count.  Repeated subsets are cached during evaluation.
    """
    normalized = dataset_name.lower()
    if eval_config.water_feature_count_protocol and normalized.startswith("water_"):
        return [
            BudgetSpec(
                index=k,
                feature_ratio=100.0 * k / d,
                num_features=k,
                protocol="WaterFeatureCount1ToD",
            )
            for k in range(1, d + 1)
        ]
    return [
        BudgetSpec(
            index=percent,
            feature_ratio=float(percent),
            num_features=choose_protocol_feature_count(d, percent),
            protocol="TopPercentage1To20",
        )
        for percent in range(1, 21)
    ]


def make_adaptive_scaler(X_train: np.ndarray, eval_config: EvaluationConfig) -> StandardScaler:
    density = float(np.count_nonzero(X_train)) / float(max(1, X_train.size))
    preserve_zeros = density <= eval_config.preserve_zeros_density_threshold
    if preserve_zeros:
        print(f"[Scaling] density={density:.4f}; preserving zeros for sparse LinearSVC.")
    return StandardScaler(with_mean=not preserve_zeros)


def serialize_feature_indices(
    indices: Sequence[int] | np.ndarray,
    one_based: bool = False,
) -> str:
    values = np.asarray(indices, dtype=np.int64)
    if one_based:
        values = values + 1
    return ";".join(map(str, values))


def select_after_removal(
    ranking: np.ndarray,
    removed_indices: Sequence[int] | np.ndarray,
    k: int,
) -> np.ndarray:
    """Remove a target set from the full ranking and refill to exactly k."""
    ranking = np.asarray(ranking, dtype=np.int64)
    removed = np.unique(np.asarray(removed_indices, dtype=np.int64))
    if removed.size == 0:
        return ranking[:k].copy()
    keep = ~np.isin(ranking, removed, assume_unique=False)
    selected = ranking[keep][:k]
    if selected.size != k:
        raise RuntimeError(
            f"Cannot refill to k={k} after removing {removed.size} features."
        )
    return selected.astype(np.int64, copy=False)


def metric_degradation(
    original_value: float,
    perturbed_value: float,
    metric_name: str,
) -> float:
    """Return a signed degradation: positive always means worse."""
    if metric_name in LOWER_IS_BETTER_METRICS:
        return float(perturbed_value - original_value)
    return float(original_value - perturbed_value)


def summarize_matched_removal(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    group_columns = [
        "Dataset",
        "RemovalTarget",
        "BudgetPercent",
        "NumFeatures",
        "Variant",
    ]
    rows: List[Dict[str, object]] = []
    for keys, subset in raw_df.groupby(group_columns, sort=False, dropna=False):
        row: Dict[str, object] = dict(zip(group_columns, keys))
        row.update(
            {
                "NumObservations": int(len(subset)),
                "NumFolds": int(subset["Fold"].nunique()),
                "MeanTargetSize": float(subset["TargetSize"].mean()),
                "MinTargetSize": int(subset["TargetSize"].min()),
                "MaxTargetSize": int(subset["TargetSize"].max()),
            }
        )
        for metric in METRIC_COLUMNS:
            row[metric] = float(subset[metric].mean())
            row[f"Std{metric}"] = float(subset[metric].std(ddof=0))
            drop_column = f"Drop{metric}"
            row[drop_column] = float(subset[drop_column].mean())
            row[f"Std{drop_column}"] = float(
                subset[drop_column].std(ddof=0)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def matched_removal_contrast_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compare targeted deletion with the fold-matched random mean."""
    if raw_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    keys = ["Dataset", "RemovalTarget", "BudgetPercent", "NumFeatures"]
    for values, subset in raw_df.groupby(keys, sort=False, dropna=False):
        target = subset[subset["Variant"] == "TargetRemoved"]
        random = subset[subset["Variant"] == "RandomRemoved"]
        if target.empty or random.empty:
            continue
        row: Dict[str, object] = dict(zip(keys, values))
        fold_records: List[Dict[str, float]] = []
        for fold in sorted(set(target["Fold"]) & set(random["Fold"])):
            target_fold = target[target["Fold"] == fold].iloc[0]
            random_fold = random[random["Fold"] == fold]
            record: Dict[str, float] = {"Fold": float(fold)}
            for metric in METRIC_COLUMNS:
                record[f"TargetDrop{metric}"] = float(
                    target_fold[f"Drop{metric}"]
                )
                record[f"RandomDrop{metric}"] = float(
                    random_fold[f"Drop{metric}"].mean()
                )
            fold_records.append(record)
        if not fold_records:
            continue
        paired = pd.DataFrame(fold_records)
        row["NumFolds"] = int(len(paired))
        for metric in METRIC_COLUMNS:
            target_column = f"TargetDrop{metric}"
            random_column = f"RandomDrop{metric}"
            excess = paired[target_column] - paired[random_column]
            row[f"MeanTargetDrop{metric}"] = float(
                paired[target_column].mean()
            )
            row[f"MeanRandomDrop{metric}"] = float(
                paired[random_column].mean()
            )
            row[f"ExcessDrop{metric}"] = float(excess.mean())
            row[f"TargetWorseThanRandomFolds{metric}"] = int(
                (excess > 0.0).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_set_stability_frame(
    dataset_name: str,
    fold_sets: Dict[int, Dict[str, Optional[np.ndarray]]],
) -> pd.DataFrame:
    """Compute auditable pairwise Jaccard values for fold feature sets."""
    rows: List[Dict[str, object]] = []
    fold_ids = sorted(fold_sets)
    for set_type in ("Necessary", "JointGroup"):
        for left_pos, fold_a in enumerate(fold_ids):
            for fold_b in fold_ids[left_pos + 1 :]:
                values_a = fold_sets[fold_a].get(set_type)
                values_b = fold_sets[fold_b].get(set_type)
                available_a = values_a is not None
                available_b = values_b is not None
                set_a = (
                    set()
                    if values_a is None
                    else set(map(int, np.asarray(values_a)))
                )
                set_b = (
                    set()
                    if values_b is None
                    else set(map(int, np.asarray(values_b)))
                )
                intersection = set_a & set_b
                union = set_a | set_b
                both_available = available_a and available_b
                jaccard = (
                    float(len(intersection) / len(union))
                    if both_available and union
                    else float("nan")
                )
                rows.append(
                    {
                        "Dataset": dataset_name,
                        "SetType": set_type,
                        "FoldA": fold_a,
                        "FoldB": fold_b,
                        "AvailableA": available_a,
                        "AvailableB": available_b,
                        "SizeA": len(set_a),
                        "SizeB": len(set_b),
                        "IntersectionSize": len(intersection),
                        "UnionSize": len(union),
                        "Jaccard": jaccard,
                        "BothEmpty": bool(
                            both_available and not set_a and not set_b
                        ),
                        "PresenceAgreement": bool(bool(set_a) == bool(set_b)),
                    }
                )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Fast data-adaptive hyperparameter search
# -----------------------------------------------------------------------------


def metric_sign(metric_name: str) -> float:
    if metric_name in {"AveragePrecision", "MicroF1", "MacroF1"}:
        return 1.0
    if metric_name in {"RankingLoss", "HammingLoss", "CoverageError", "OneError"}:
        return -1.0
    raise ValueError(f"Unknown tuning metric: {metric_name}")


def _fit_and_score_ratio_config(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val_target: np.ndarray,
    lambda_ratio: float,
    gamma_ratio: float,
    lambda_max: float,
    design_lipschitz: float,
    b0: np.ndarray,
    selector_config: SelectorConfig,
    eval_config: EvaluationConfig,
    initial_fit: Optional[FitResult] = None,
) -> Tuple[Dict[str, float], FitResult]:
    lam = float(lambda_ratio * lambda_max)
    gamma = float(gamma_ratio * design_lipschitz)
    selector = ConvexPartialLabelSelector(lam, gamma, selector_config, design_lipschitz)
    fit = selector.fit(
        X_train,
        Y_train,
        initial_W=None if initial_fit is None else initial_fit.W,
        initial_b=b0 if initial_fit is None else initial_fit.b,
        max_iter=selector_config.tuning_max_iter,
        certify=False,
    )
    importance = np.linalg.norm(fit.W, axis=1)
    ranking = np.argsort(-importance, kind="stable")
    k = choose_protocol_feature_count(
        X_train.shape[1],
        20,
    )
    metrics, _ = evaluate_feature_subset(
        X_train,
        Y_train,
        X_val,
        Y_val_target,
        ranking[:k],
        eval_config,
    )
    row: Dict[str, float] = {
        "LambdaRatio": float(lambda_ratio),
        "GammaRatio": float(gamma_ratio),
        "Lambda": lam,
        "Gamma": gamma,
        "LambdaMax": lambda_max,
        "DesignLipschitz": design_lipschitz,
        "NumNonzeroRows": int(np.sum(importance > 1e-10)),
        "Objective": fit.objective,
        "FitIterations": fit.n_iter,
        "FitSeconds": fit.elapsed_seconds,
        "TuningNumFeatures": k,
    }
    row.update(metrics)
    row["SelectionScore"] = metric_sign(eval_config.tuning_metric) * metrics[eval_config.tuning_metric]
    return row, fit


def fast_hyperparameter_search(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val_target: np.ndarray,
    selector_config: SelectorConfig,
    eval_config: EvaluationConfig,
) -> Tuple[float, float, pd.DataFrame]:
    print("[Tuning] estimating lambda_max and design curvature...")
    design_lipschitz = estimate_design_lipschitz(
        X_train,
        n_iter=selector_config.power_iterations,
        random_state=selector_config.random_state,
    )
    lambda_max, b0 = estimate_lambda_max(
        X_train, Y_train, selector_config.intercept_iterations
    )
    print(
        f"[Tuning] lambda_max={lambda_max:.4e}, "
        f"design_lipschitz={design_lipschitz:.4e}"
    )

    rows: List[Dict[str, float]] = []
    fits: Dict[Tuple[float, float], FitResult] = {}
    lambda_ratios = sorted(set(map(float, selector_config.lambda_ratio_grid)), reverse=True)
    gamma_ratios = sorted(set(map(float, selector_config.gamma_ratio_grid)))

    total_initial = len(lambda_ratios) * len(gamma_ratios)
    counter = 0
    for gamma_ratio in gamma_ratios:
        warm: Optional[FitResult] = None
        for lambda_ratio in lambda_ratios:
            counter += 1
            print(
                f"[Tuning] initial {counter}/{total_initial}: "
                f"lambda_ratio={lambda_ratio:g}, gamma_ratio={gamma_ratio:g}"
            )
            row, fit = _fit_and_score_ratio_config(
                X_train,
                Y_train,
                X_val,
                Y_val_target,
                lambda_ratio,
                gamma_ratio,
                lambda_max,
                design_lipschitz,
                b0,
                selector_config,
                eval_config,
                initial_fit=warm,
            )
            rows.append(row)
            fits[(lambda_ratio, gamma_ratio)] = fit
            warm = fit

    def current_best() -> Dict[str, float]:
        ordered = sorted(
            rows,
            key=lambda r: (
                -float(r["SelectionScore"]),
                int(r["NumNonzeroRows"]),
            ),
        )
        return ordered[0]

    # Add only direct neighbours of a boundary optimum; never form another
    # full Cartesian product.
    for expansion_round in range(selector_config.boundary_expansion_rounds):
        best = current_best()
        lr = float(best["LambdaRatio"])
        gr = float(best["GammaRatio"])
        known_lrs = sorted({float(r["LambdaRatio"]) for r in rows})
        known_grs = sorted({float(r["GammaRatio"]) for r in rows})
        new_lrs: List[float] = []
        new_grs: List[float] = []
        if math.isclose(lr, min(known_lrs)):
            new_lrs.append(lr / 10.0)
        if (
            math.isclose(lr, max(known_lrs))
            and lr < selector_config.lambda_ratio_expansion_max
        ):
            new_lrs.append(
                min(selector_config.lambda_ratio_expansion_max, lr * 5.0)
            )
        if math.isclose(gr, min(known_grs)):
            new_grs.append(gr / 10.0)
        if math.isclose(gr, max(known_grs)):
            new_grs.append(gr * 10.0)

        candidates: List[Tuple[float, float]] = []
        candidates.extend((x, gr) for x in new_lrs)
        candidates.extend((lr, x) for x in new_grs)
        candidates.extend((x, y) for x in new_lrs for y in new_grs)
        candidates = [c for c in candidates if c not in fits]
        if not candidates:
            break

        best_fit = fits[(lr, gr)]
        for idx, (new_lr, new_gr) in enumerate(candidates, start=1):
            print(
                f"[Tuning] expansion {expansion_round + 1}, {idx}/{len(candidates)}: "
                f"lambda_ratio={new_lr:g}, gamma_ratio={new_gr:g}"
            )
            row, fit = _fit_and_score_ratio_config(
                X_train,
                Y_train,
                X_val,
                Y_val_target,
                new_lr,
                new_gr,
                lambda_max,
                design_lipschitz,
                b0,
                selector_config,
                eval_config,
                initial_fit=best_fit,
            )
            rows.append(row)
            fits[(new_lr, new_gr)] = fit

    search_df = pd.DataFrame(rows).sort_values(
        ["SelectionScore", "NumNonzeroRows"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    best_lambda_ratio = float(search_df.iloc[0]["LambdaRatio"])
    best_gamma_ratio = float(search_df.iloc[0]["GammaRatio"])
    print(
        f"[Tuning] best lambda_ratio={best_lambda_ratio:g}, "
        f"gamma_ratio={best_gamma_ratio:g}, "
        f"{eval_config.tuning_metric}={search_df.iloc[0][eval_config.tuning_metric]:.6f}"
    )
    return best_lambda_ratio, best_gamma_ratio, search_df


# -----------------------------------------------------------------------------
# Cross-validation experiment and output
# -----------------------------------------------------------------------------


def _safe_name(name: str) -> str:
    invalid = '<>:"/\\|?*'
    value = "".join("_" if c in invalid else c for c in name).strip()
    return value or "dataset"


def _fit_selector_for_fold(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    best_lambda_ratio: float,
    best_gamma_ratio: float,
    selector_config: SelectorConfig,
) -> Tuple[
    ConvexPartialLabelSelector,
    FitResult,
    EpsilonReference,
    float,
    float,
    float,
    np.ndarray,
]:
    design_lipschitz = estimate_design_lipschitz(
        X_train,
        n_iter=selector_config.power_iterations,
        random_state=selector_config.random_state,
    )
    lambda_max, b0 = estimate_lambda_max(
        X_train, Y_train, selector_config.intercept_iterations
    )
    lam = best_lambda_ratio * lambda_max
    gamma = best_gamma_ratio * design_lipschitz
    selector = ConvexPartialLabelSelector(lam, gamma, selector_config, design_lipschitz)
    fit = fit_with_adaptive_certificate(
        selector,
        X_train,
        Y_train,
        initial_b=b0,
        initial_max_iter=selector_config.max_iter,
    )
    intercept_fit = fit_intercept_only_reference(
        Y_train,
        gamma,
        fit,
        selector_config,
        initial_b=b0,
    )
    epsilon_reference = build_epsilon_reference(
        fit,
        intercept_fit,
        selector_config,
    )

    # The initial fit used the total objective only as a numerical scale.
    # Continue it once if the new, feature-relative epsilon is materially
    # smaller, then rebuild the safe interval.
    feature_scale = max(
        epsilon_reference.epsilon_lower,
        epsilon_reference.epsilon,
        EPS,
    )
    if (
        selector_config.epsilon_mode.lower().strip() == "feature_gain"
        and fit.optimization_gap_upper / feature_scale
        > selector_config.certificate_gap_ratio_target
    ):
        previous_fit = fit
        continued_fit = fit_with_adaptive_certificate(
            selector,
            X_train,
            Y_train,
            initial_W=fit.W,
            initial_b=fit.b,
            initial_max_iter=selector_config.adaptive_refinement_max_iter,
            epsilon_reference=feature_scale,
        )
        fit = replace(
            continued_fit,
            n_iter=previous_fit.n_iter + continued_fit.n_iter,
            elapsed_seconds=(
                previous_fit.elapsed_seconds
                + continued_fit.elapsed_seconds
            ),
            adaptive_refinement_rounds=(
                previous_fit.adaptive_refinement_rounds
                + continued_fit.adaptive_refinement_rounds
            ),
        )
        previous_intercept_fit = intercept_fit
        continued_intercept_fit = fit_intercept_only_reference(
            Y_train,
            gamma,
            fit,
            selector_config,
            initial_b=intercept_fit.b,
        )
        intercept_fit = replace(
            continued_intercept_fit,
            n_iter=(
                previous_intercept_fit.n_iter
                + continued_intercept_fit.n_iter
            ),
            elapsed_seconds=(
                previous_intercept_fit.elapsed_seconds
                + continued_intercept_fit.elapsed_seconds
            ),
        )
        epsilon_reference = build_epsilon_reference(
            fit,
            intercept_fit,
            selector_config,
        )
    final_reference_scale = max(
        epsilon_reference.epsilon_lower,
        epsilon_reference.epsilon,
        EPS,
    )
    final_gap_ratio = fit.optimization_gap_upper / final_reference_scale
    fit = replace(
        fit,
        optimization_gap_ratio=float(final_gap_ratio),
        certificate_converged=bool(
            final_gap_ratio
            <= selector_config.certificate_gap_ratio_target
        ),
    )
    return (
        selector,
        fit,
        epsilon_reference,
        lam,
        gamma,
        lambda_max,
        b0,
    )


def _summarize_fold_results(
    fold_df: pd.DataFrame,
    best_lambda_ratio: float,
    best_gamma_ratio: float,
    eval_config: EvaluationConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    keys: List[Tuple[str, Optional[int], float, str]] = [
        ("Necessary", None, float("nan"), "Necessary")
    ]
    ranked_keys = (
        fold_df[fold_df["SelectionMode"] == "RankedProtocol"][
            ["BudgetIndex", "FeatureRatio(%)", "BudgetProtocol"]
        ]
        .drop_duplicates()
        .sort_values("BudgetIndex", kind="stable")
    )
    keys.extend(
        (
            "RankedProtocol",
            int(item["BudgetIndex"]),
            float(item["FeatureRatio(%)"]),
            str(item["BudgetProtocol"]),
        )
        for _, item in ranked_keys.iterrows()
    )
    for mode, budget_index, ratio, budget_protocol in keys:
        if budget_index is None:
            subset = fold_df[fold_df["SelectionMode"] == mode]
        else:
            subset = fold_df[
                (fold_df["SelectionMode"] == mode)
                & (fold_df["BudgetIndex"] == budget_index)
            ]
        counts = subset["NumFeatures"].astype(float)
        row: Dict[str, object] = {
            "SelectionMode": mode,
            "BudgetIndex": np.nan if budget_index is None else budget_index,
            "BudgetProtocol": budget_protocol,
            "FeatureRatio(%)": ratio,
            "NumFeatures": float(counts.mean()),
            "StdNumFeatures": float(counts.std(ddof=0)),
            "MinNumFeatures": int(counts.min()),
            "MaxNumFeatures": int(counts.max()),
            "UsedEvalFolds": int(len(subset)),
            "GridSearchFold": eval_config.tuning_fold_index + 1,
            "best_lambda_ratio": best_lambda_ratio,
            "best_gamma_ratio": best_gamma_ratio,
            "MeanNumNecessary": float(subset["NumNecessary"].mean()),
            "MeanNumUnresolved": float(subset["NumUnresolved"].mean()),
            "EpsilonMode": str(subset["EpsilonMode"].iloc[0]),
            "MeanEpsilon": float(subset["Epsilon"].mean()),
            "MeanEpsilonLower": float(subset["EpsilonLower"].mean()),
            "MeanEpsilonUpper": float(subset["EpsilonUpper"].mean()),
            "MeanFeatureGain": float(subset["FeatureGain"].mean()),
            "MeanFeatureGainLower": float(
                subset["FeatureGainLower"].mean()
            ),
            "MeanFeatureGainUpper": float(
                subset["FeatureGainUpper"].mean()
            ),
            "AllInterceptOnlyCertificatesConverged": bool(
                subset["InterceptOnlyCertificateConverged"].astype(bool).all()
            ),
            "MeanInterceptOnlyIterations": float(
                subset["InterceptOnlyIterations"].mean()
            ),
            "MeanInterceptOnlySeconds": float(
                subset["InterceptOnlySeconds"].mean()
            ),
            "MeanOptimizationGapRatio": float(subset["OptimizationGapRatio"].mean()),
            "MaxOptimizationGapRatio": float(subset["OptimizationGapRatio"].max()),
            "AllBaselineCertificatesConverged": bool(
                subset["CertificateConverged"].astype(bool).all()
            ),
            "MeanAdaptiveRefinementRounds": float(
                subset["AdaptiveRefinementRounds"].mean()
            ),
            "MeanExactRefits": float(subset["ExactRefits"].mean()),
            "MeanExactRefitsConverged": float(
                subset["ExactRefitsConverged"].mean()
            ),
            "AllExactRefitCertificatesConverged": bool(
                (
                    subset["ExactRefitsConverged"].astype(int)
                    == subset["ExactRefits"].astype(int)
                ).all()
            ),
            "MaxOffOptimizationGapRatio": float(
                subset["MaxOffOptimizationGapRatio"].max()
            ),
            "MeanSelectorSeconds": float(subset["SelectorSeconds"].mean()),
            "MeanNecessitySeconds": float(subset["NecessitySeconds"].mean()),
        }
        for metric in METRIC_COLUMNS:
            row[metric] = float(subset[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_dataset_experiment(
    mat_path: str,
    output_dir: str,
    selector_config: Optional[SelectorConfig] = None,
    eval_config: Optional[EvaluationConfig] = None,
    x_key: str = "data",
    partial_key: str = "partial_labels",
    target_key: str = "target",
) -> pd.DataFrame:
    selector_config = selector_config or SelectorConfig()
    eval_config = eval_config or EvaluationConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X, Y, Y_target = load_mat_dataset(mat_path, x_key, partial_key, target_key)
    dataset_name = Path(mat_path).stem
    safe_name = _safe_name(dataset_name)
    budget_specs = build_protocol_budgets(
        dataset_name,
        X.shape[1],
        eval_config,
    )
    print(summarize_dataset(dataset_name, X, Y, Y_target))
    print(
        f"[{dataset_name}] budget protocol={budget_specs[0].protocol}, "
        f"positions={len(budget_specs)}, "
        f"feature-count range={budget_specs[0].num_features}--"
        f"{budget_specs[-1].num_features}"
    )

    folds = list(
        KFold(
            n_splits=eval_config.n_splits,
            shuffle=True,
            random_state=eval_config.random_state,
        ).split(X)
    )
    tuning_fold = eval_config.tuning_fold_index
    if not 0 <= tuning_fold < len(folds):
        raise ValueError("tuning_fold_index is outside the KFold range.")

    _, validation_idx = folds[tuning_fold]
    tuning_train_idx = np.setdiff1d(np.arange(X.shape[0]), validation_idx)
    X_tune_train_raw = X[tuning_train_idx]
    scaler = make_adaptive_scaler(X_tune_train_raw, eval_config)
    X_tune_train = scaler.fit_transform(X_tune_train_raw).astype(FLOATX, copy=False)
    X_tune_val = scaler.transform(X[validation_idx]).astype(FLOATX, copy=False)
    best_lr, best_gr, search_df = fast_hyperparameter_search(
        X_tune_train,
        Y[tuning_train_idx],
        X_tune_val,
        Y_target[validation_idx],
        selector_config,
        eval_config,
    )
    search_df.insert(0, "Dataset", dataset_name)
    search_df.to_csv(
        output_path / f"{safe_name}_hyperparameter_search.csv",
        index=False,
        encoding="utf-8-sig",
    )
    del X_tune_train_raw, X_tune_train, X_tune_val

    fold_rows: List[Dict[str, object]] = []
    fold_selection_rows: List[Dict[str, object]] = []
    fold_group_frames: List[pd.DataFrame] = []
    matched_removal_rows: List[Dict[str, object]] = []
    fold_feature_sets: Dict[int, Dict[str, Optional[np.ndarray]]] = {}
    for fold_id, (_, test_idx) in enumerate(folds):
        if fold_id == tuning_fold:
            continue
        print(f"[{dataset_name}] evaluation fold {fold_id + 1}/{len(folds)}")
        train_idx = np.setdiff1d(np.arange(X.shape[0]), test_idx)
        X_train_raw = X[train_idx]
        scaler = make_adaptive_scaler(X_train_raw, eval_config)
        X_train = scaler.fit_transform(X_train_raw).astype(FLOATX, copy=False)
        X_test = scaler.transform(X[test_idx]).astype(FLOATX, copy=False)
        Y_train = Y[train_idx]
        Y_test_target = Y_target[test_idx]

        (
            selector,
            fit,
            epsilon_reference,
            lam,
            gamma,
            lambda_max,
            _,
        ) = _fit_selector_for_fold(
            X_train,
            Y_train,
            best_lr,
            best_gr,
            selector_config,
        )
        necessity = certify_necessary_features(
            X_train,
            Y_train,
            selector,
            fit,
            selector_config,
            epsilon_reference,
        )
        ranking, _ = necessity_first_ranking(fit.W, necessity.necessary)
        necessary_idx = np.flatnonzero(necessity.necessary)
        unresolved_idx = np.flatnonzero(necessity.unresolved)
        n_necessary = int(len(necessary_idx))
        n_unresolved = int(len(unresolved_idx))
        print(
            f"[{dataset_name}] fold={fold_id + 1}, necessary={n_necessary}, "
            f"unresolved={n_unresolved}, mode={necessity.mode_used}, "
            f"gap/epsilon={fit.optimization_gap_ratio:.3e}, "
            f"certificate={'OK' if fit.certificate_converged else 'FAILED'}, "
            f"selector={fit.elapsed_seconds:.2f}s"
        )

        # Discover a jointly necessary prefix inside this training fold only.
        # It is used for held-out removal validation only when no singleton
        # feature has been certified in that fold.
        joint_group_idx: Optional[np.ndarray] = None
        fold_group_seconds = 0.0
        fold_group_status = (
            "SkippedSingletonNecessary"
            if n_necessary > 0
            else "NotRun"
        )
        if (
            eval_config.enable_fold_group_diagnostic
            and n_necessary == 0
        ):
            print(
                f"[{dataset_name}] fold={fold_id + 1}, searching for a "
                "training-only jointly necessary prefix..."
            )
            fold_group_df = cumulative_group_diagnostic(
                X_train,
                Y_train,
                selector,
                fit,
                necessity,
                ranking,
            )
            fold_group_df.insert(0, "Fold", fold_id + 1)
            fold_group_df.insert(0, "Dataset", dataset_name)
            fold_group_frames.append(fold_group_df)
            fold_group_seconds = float(
                fold_group_df["ElapsedSeconds"].sum()
            )
            certified_rows = fold_group_df[
                fold_group_df["FirstCertifiedNecessaryPrefix"].astype(bool)
            ]
            if not certified_rows.empty:
                group_size = int(certified_rows.iloc[0]["GroupSize"])
                joint_group_idx = ranking[:group_size].copy()
                fold_group_status = "Certified"
            else:
                fold_group_status = "NoCertifiedGroupWithinSearchLimit"

        matched_target_type: Optional[str]
        matched_target_idx: Optional[np.ndarray]
        if n_necessary > 0:
            matched_target_type = "SingletonNecessary"
            matched_target_idx = necessary_idx.copy()
        elif joint_group_idx is not None:
            matched_target_type = "JointGroup"
            matched_target_idx = joint_group_idx.copy()
        else:
            matched_target_type = None
            matched_target_idx = None

        fold_number = fold_id + 1
        fold_feature_sets[fold_number] = {
            "Necessary": necessary_idx.copy(),
            "JointGroup": (
                None
                if joint_group_idx is None
                else joint_group_idx.copy()
            ),
        }
        fold_selection_rows.append(
            {
                "Dataset": dataset_name,
                "Fold": fold_number,
                "NumNecessary": n_necessary,
                "NecessaryIndices0": serialize_feature_indices(
                    necessary_idx
                ),
                "NecessaryIndices1": serialize_feature_indices(
                    necessary_idx,
                    one_based=True,
                ),
                "NumUnresolved": n_unresolved,
                "UnresolvedIndices0": serialize_feature_indices(
                    unresolved_idx
                ),
                "UnresolvedIndices1": serialize_feature_indices(
                    unresolved_idx,
                    one_based=True,
                ),
                "FoldGroupStatus": fold_group_status,
                "JointGroupSize": (
                    0
                    if joint_group_idx is None
                    else int(joint_group_idx.size)
                ),
                "JointGroupIndices0": (
                    ""
                    if joint_group_idx is None
                    else serialize_feature_indices(joint_group_idx)
                ),
                "JointGroupIndices1": (
                    ""
                    if joint_group_idx is None
                    else serialize_feature_indices(
                        joint_group_idx,
                        one_based=True,
                    )
                ),
                "MatchedRemovalTarget": (
                    "None"
                    if matched_target_type is None
                    else matched_target_type
                ),
                "MatchedRemovalTargetSize": (
                    0
                    if matched_target_idx is None
                    else int(matched_target_idx.size)
                ),
                "FoldGroupSeconds": fold_group_seconds,
                "MatchedRemovalSeconds": 0.0,
            }
        )

        common: Dict[str, object] = {
            "Dataset": dataset_name,
            "Fold": fold_number,
            "NumNecessary": n_necessary,
            "NumUnresolved": n_unresolved,
            "NecessityMode": necessity.mode_used,
            "ExactRefits": necessity.exact_refits,
            "ExactRefitsConverged": necessity.exact_refits_converged,
            "MaxOffOptimizationGapRatio": necessity.max_off_optimization_gap_ratio,
            "Epsilon": necessity.epsilon,
            "EpsilonLower": necessity.epsilon_lower,
            "EpsilonUpper": necessity.epsilon_upper,
            "EpsilonMode": necessity.epsilon_mode,
            "FeatureGain": necessity.feature_gain,
            "FeatureGainLower": necessity.feature_gain_lower,
            "FeatureGainUpper": necessity.feature_gain_upper,
            "InterceptOnlyObjective": necessity.intercept_objective,
            "InterceptOnlyOptimizationGapUpper": (
                necessity.intercept_optimization_gap_upper
            ),
            "InterceptOnlyCertificateConverged": (
                necessity.intercept_certificate_converged
            ),
            "InterceptOnlyIterations": necessity.intercept_n_iter,
            "InterceptOnlySeconds": necessity.intercept_elapsed_seconds,
            "Lambda": lam,
            "Gamma": gamma,
            "LambdaMax": lambda_max,
            "SelectorObjective": fit.objective,
            "OptimizationGapUpper": fit.optimization_gap_upper,
            "OptimizationGapRatio": fit.optimization_gap_ratio,
            "CertificateConverged": fit.certificate_converged,
            "AdaptiveRefinementRounds": fit.adaptive_refinement_rounds,
            "SelectorIterations": fit.n_iter,
            "SelectorSeconds": fit.elapsed_seconds,
            "NecessitySeconds": necessity.elapsed_seconds,
        }

        # Adjacent percentages can round to the same feature count on
        # low-dimensional data; cache those repeated SVM evaluations.
        evaluation_cache: Dict[Tuple[int, ...], Tuple[Dict[str, float], str]] = {}

        def cached_evaluate(indices: np.ndarray) -> Tuple[Dict[str, float], str]:
            key = tuple(map(int, indices))
            if key not in evaluation_cache:
                evaluation_cache[key] = evaluate_feature_subset(
                    X_train,
                    Y_train,
                    X_test,
                    Y_test_target,
                    indices,
                    eval_config,
                )
            return evaluation_cache[key]

        metrics, classifier_mode = cached_evaluate(necessary_idx)
        row = dict(common)
        row.update(
            {
                "SelectionMode": "Necessary",
                "BudgetIndex": np.nan,
                "BudgetProtocol": "Necessary",
                "FeatureRatio(%)": np.nan,
                "NumFeatures": n_necessary,
                "ClassifierMode": classifier_mode,
            }
        )
        row.update(metrics)
        fold_rows.append(row)

        for budget in budget_specs:
            k = budget.num_features
            selected = ranking[:k]
            metrics, classifier_mode = cached_evaluate(selected)
            row = dict(common)
            row.update(
                {
                    "SelectionMode": "RankedProtocol",
                    "BudgetIndex": budget.index,
                    "BudgetProtocol": budget.protocol,
                    "FeatureRatio(%)": budget.feature_ratio,
                    "NumFeatures": k,
                    "ClassifierMode": classifier_mode,
                }
            )
            row.update(metrics)
            fold_rows.append(row)

        if (
            eval_config.enable_matched_removal
            and matched_target_type is not None
            and matched_target_idx is not None
            and matched_target_idx.size > 0
        ):
            matched_removal_start = time.perf_counter()
            target_size = int(matched_target_idx.size)
            target_indices0 = serialize_feature_indices(
                matched_target_idx
            )
            target_indices1 = serialize_feature_indices(
                matched_target_idx,
                one_based=True,
            )
            for budget_percent in eval_config.matched_removal_budgets:
                if not 0 < int(budget_percent) <= 100:
                    raise ValueError(
                        "matched_removal_budgets must contain percentages "
                        "in [1, 100]."
                    )
                k = choose_protocol_feature_count(
                    X.shape[1],
                    int(budget_percent),
                )
                if target_size > k:
                    if selector_config.verbose:
                        print(
                            f"[MatchedRemoval] {dataset_name}, "
                            f"fold={fold_number}, budget={budget_percent}%, "
                            f"skipped because target_size={target_size} > k={k}."
                        )
                    continue
                original_indices = ranking[:k].copy()
                if not np.all(np.isin(matched_target_idx, original_indices)):
                    raise RuntimeError(
                        "The certified removal target is not fully contained "
                        "in the original top-k subset."
                    )
                random_pool = original_indices[
                    ~np.isin(original_indices, matched_target_idx)
                ]
                if random_pool.size < target_size:
                    if selector_config.verbose:
                        print(
                            f"[MatchedRemoval] {dataset_name}, "
                            f"fold={fold_number}, budget={budget_percent}%, "
                            "skipped because there are too few non-target "
                            "top-k features for a disjoint random control."
                        )
                    continue
                original_metrics, original_classifier = cached_evaluate(
                    original_indices
                )

                def append_matched_row(
                    variant: str,
                    repeat: int,
                    selected_indices: np.ndarray,
                    removed_indices: np.ndarray,
                    selected_metrics: Dict[str, float],
                    selected_classifier: str,
                ) -> None:
                    matched_row: Dict[str, object] = {
                        "Dataset": dataset_name,
                        "Fold": fold_number,
                        "RemovalTarget": matched_target_type,
                        "TargetSize": target_size,
                        "TargetIndices0": target_indices0,
                        "TargetIndices1": target_indices1,
                        "BudgetPercent": int(budget_percent),
                        "NumFeatures": k,
                        "Variant": variant,
                        "RandomRepeat": repeat,
                        "RemovedIndices0": serialize_feature_indices(
                            np.sort(removed_indices)
                        ),
                        "RemovedIndices1": serialize_feature_indices(
                            np.sort(removed_indices),
                            one_based=True,
                        ),
                        "SelectedIndices0": serialize_feature_indices(
                            selected_indices
                        ),
                        "ClassifierMode": selected_classifier,
                    }
                    for metric in METRIC_COLUMNS:
                        matched_row[metric] = float(
                            selected_metrics[metric]
                        )
                        matched_row[f"Drop{metric}"] = metric_degradation(
                            original_metrics[metric],
                            selected_metrics[metric],
                            metric,
                        )
                    matched_removal_rows.append(matched_row)

                append_matched_row(
                    "Original",
                    0,
                    original_indices,
                    np.empty(0, dtype=np.int64),
                    original_metrics,
                    original_classifier,
                )

                target_removed_indices = select_after_removal(
                    ranking,
                    matched_target_idx,
                    k,
                )
                target_metrics, target_classifier = cached_evaluate(
                    target_removed_indices
                )
                append_matched_row(
                    "TargetRemoved",
                    0,
                    target_removed_indices,
                    matched_target_idx,
                    target_metrics,
                    target_classifier,
                )

                target_code = (
                    1
                    if matched_target_type == "SingletonNecessary"
                    else 2
                )
                random_seed = (
                    int(eval_config.random_state)
                    + fold_number * 1_000_003
                    + int(budget_percent) * 10_009
                    + target_code * 1_009
                )
                rng = np.random.default_rng(random_seed)
                for repeat in range(
                    1,
                    max(
                        0,
                        int(eval_config.matched_removal_random_repeats),
                    )
                    + 1,
                ):
                    random_removed = np.asarray(
                        rng.choice(
                            random_pool,
                            size=target_size,
                            replace=False,
                        ),
                        dtype=np.int64,
                    )
                    random_selected = select_after_removal(
                        ranking,
                        random_removed,
                        k,
                    )
                    random_metrics, random_classifier = cached_evaluate(
                        random_selected
                    )
                    append_matched_row(
                        "RandomRemoved",
                        repeat,
                        random_selected,
                        random_removed,
                        random_metrics,
                        random_classifier,
                    )
            fold_selection_rows[-1]["MatchedRemovalSeconds"] = (
                time.perf_counter() - matched_removal_start
            )

        del X_train_raw, X_train, X_test

    fold_df = pd.DataFrame(fold_rows)
    summary_df = _summarize_fold_results(fold_df, best_lr, best_gr, eval_config)
    summary_df.insert(0, "Dataset", dataset_name)
    fold_df.to_csv(
        output_path / f"{safe_name}_fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        output_path / f"{safe_name}_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_selection_df = pd.DataFrame(fold_selection_rows)
    fold_selection_df.to_csv(
        output_path / f"{safe_name}_fold_feature_sets.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stability_df = pairwise_set_stability_frame(
        dataset_name,
        fold_feature_sets,
    )
    stability_df.to_csv(
        output_path / f"{safe_name}_fold_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stability_summary_rows: List[Dict[str, object]] = []
    for set_type, subset in stability_df.groupby("SetType", sort=False):
        valid = subset["Jaccard"].notna()
        stability_summary_rows.append(
            {
                "Dataset": dataset_name,
                "SetType": set_type,
                "NumPairs": int(len(subset)),
                "NumPairsWithDefinedJaccard": int(valid.sum()),
                "MeanJaccard": (
                    float(subset.loc[valid, "Jaccard"].mean())
                    if np.any(valid)
                    else float("nan")
                ),
                "MinJaccard": (
                    float(subset.loc[valid, "Jaccard"].min())
                    if np.any(valid)
                    else float("nan")
                ),
                "MaxJaccard": (
                    float(subset.loc[valid, "Jaccard"].max())
                    if np.any(valid)
                    else float("nan")
                ),
                "BothEmptyPairs": int(subset["BothEmpty"].sum()),
                "PresenceAgreementRate": float(
                    subset["PresenceAgreement"].mean()
                ),
            }
        )
    pd.DataFrame(stability_summary_rows).to_csv(
        output_path / f"{safe_name}_fold_stability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if fold_group_frames:
        pd.concat(fold_group_frames, ignore_index=True).to_csv(
            output_path / f"{safe_name}_fold_group_diagnostic.csv",
            index=False,
            encoding="utf-8-sig",
        )

    matched_raw_df = pd.DataFrame(matched_removal_rows)
    if not matched_raw_df.empty:
        matched_raw_df.to_csv(
            output_path / f"{safe_name}_matched_removal_raw.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summarize_matched_removal(matched_raw_df).to_csv(
            output_path / f"{safe_name}_matched_removal_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        matched_removal_contrast_frame(matched_raw_df).to_csv(
            output_path / f"{safe_name}_matched_removal_contrast.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        print(
            f"[{dataset_name}] no certified fold-level removal target; "
            "matched-removal files were not created."
        )

    if eval_config.fit_full_after_cv:
        print(f"[{dataset_name}] fitting final selector on all samples...")
        scaler_full = make_adaptive_scaler(X, eval_config)
        X_full = scaler_full.fit_transform(X).astype(FLOATX, copy=False)
        (
            selector,
            fit,
            epsilon_reference,
            lam,
            gamma,
            lambda_max,
            _,
        ) = _fit_selector_for_fold(
            X_full,
            Y,
            best_lr,
            best_gr,
            selector_config,
        )
        necessity = certify_necessary_features(
            X_full,
            Y,
            selector,
            fit,
            selector_config,
            epsilon_reference,
        )
        detail_df = feature_detail_frame(fit, necessity)
        detail_df.insert(0, "Dataset", dataset_name)
        detail_df.to_csv(
            output_path / f"{safe_name}_feature_details.csv",
            index=False,
            encoding="utf-8-sig",
        )
        ranking, importance = necessity_first_ranking(fit.W, necessity.necessary)
        should_run_group_diagnostic = (
            selector_config.enable_group_diagnostic
            and (
                not selector_config.group_diagnostic_only_when_no_singleton
                or not np.any(necessity.necessary)
            )
        )
        if should_run_group_diagnostic:
            print(
                f"[{dataset_name}] running cumulative jointly-necessary "
                "group diagnostic..."
            )
            group_df = cumulative_group_diagnostic(
                X_full,
                Y,
                selector,
                fit,
                necessity,
                ranking,
            )
            group_df.insert(0, "Dataset", dataset_name)
            group_df.to_csv(
                output_path / f"{safe_name}_group_diagnostic.csv",
                index=False,
                encoding="utf-8-sig",
            )
        elif (
            selector_config.enable_group_diagnostic
            and selector_config.group_diagnostic_only_when_no_singleton
        ):
            print(
                f"[{dataset_name}] group diagnostic skipped because the "
                "full-data model already has singleton necessary features."
            )
        np.savez_compressed(
            output_path / f"{safe_name}_model.npz",
            W=fit.W,
            b=fit.b,
            scaler_mean=(
                np.zeros(X.shape[1], dtype=np.float64)
                if scaler_full.mean_ is None
                else scaler_full.mean_
            ),
            scaler_scale=scaler_full.scale_,
            scaler_with_mean=np.asarray(
                [bool(scaler_full.with_mean)],
                dtype=np.bool_,
            ),
            necessary=necessity.necessary,
            unresolved=necessity.unresolved,
            ranking=ranking,
            importance=importance,
            off_gap_lower=necessity.off_gap_lower,
            off_gap_upper=necessity.off_gap_upper,
            epsilon=np.asarray([necessity.epsilon]),
            epsilon_lower=np.asarray([necessity.epsilon_lower]),
            epsilon_upper=np.asarray([necessity.epsilon_upper]),
            epsilon_mode=np.asarray([necessity.epsilon_mode]),
            feature_gain=np.asarray([necessity.feature_gain]),
            feature_gain_lower=np.asarray([necessity.feature_gain_lower]),
            feature_gain_upper=np.asarray([necessity.feature_gain_upper]),
            intercept_only_objective=np.asarray(
                [necessity.intercept_objective]
            ),
            intercept_only_optimization_gap_upper=np.asarray(
                [necessity.intercept_optimization_gap_upper]
            ),
            intercept_only_certificate_converged=np.asarray(
                [necessity.intercept_certificate_converged]
            ),
            intercept_only_iterations=np.asarray(
                [necessity.intercept_n_iter]
            ),
            intercept_only_seconds=np.asarray(
                [necessity.intercept_elapsed_seconds]
            ),
            lambda_value=np.asarray([lam]),
            gamma_value=np.asarray([gamma]),
            lambda_max=np.asarray([lambda_max]),
            best_lambda_ratio=np.asarray([best_lr]),
            best_gamma_ratio=np.asarray([best_gr]),
            optimization_gap_ratio=np.asarray([fit.optimization_gap_ratio]),
            certificate_converged=np.asarray([fit.certificate_converged]),
            adaptive_refinement_rounds=np.asarray([fit.adaptive_refinement_rounds]),
            exact_refits_converged=np.asarray([necessity.exact_refits_converged]),
            max_off_optimization_gap_ratio=np.asarray(
                [necessity.max_off_optimization_gap_ratio]
            ),
        )

    print(f"[{dataset_name}] results saved to {output_path.resolve()}")
    return summary_df


def run_multiple_datasets(
    mat_paths: Sequence[str],
    output_dir: str,
    selector_config: Optional[SelectorConfig] = None,
    eval_config: Optional[EvaluationConfig] = None,
) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    combined: List[pd.DataFrame] = []
    combined_matched: List[pd.DataFrame] = []
    combined_contrast: List[pd.DataFrame] = []
    combined_stability: List[pd.DataFrame] = []
    for idx, mat_path in enumerate(mat_paths, start=1):
        print(f"[Batch] dataset {idx}/{len(mat_paths)}: {mat_path}")
        summary = run_dataset_experiment(
            mat_path,
            output_dir,
            selector_config=selector_config,
            eval_config=eval_config,
        )
        name = Path(mat_path).stem
        results[name] = summary
        combined.append(summary)
        safe_name = _safe_name(name)
        matched_path = (
            Path(output_dir)
            / f"{safe_name}_matched_removal_summary.csv"
        )
        contrast_path = (
            Path(output_dir)
            / f"{safe_name}_matched_removal_contrast.csv"
        )
        stability_path = (
            Path(output_dir)
            / f"{safe_name}_fold_stability_summary.csv"
        )
        if matched_path.exists():
            combined_matched.append(pd.read_csv(matched_path))
        if contrast_path.exists():
            combined_contrast.append(pd.read_csv(contrast_path))
        if stability_path.exists():
            combined_stability.append(pd.read_csv(stability_path))
    if combined:
        pd.concat(combined, ignore_index=True).to_csv(
            Path(output_dir) / "all_datasets_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if combined_matched:
        pd.concat(combined_matched, ignore_index=True).to_csv(
            Path(output_dir)
            / "all_datasets_matched_removal_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if combined_contrast:
        pd.concat(combined_contrast, ignore_index=True).to_csv(
            Path(output_dir)
            / "all_datasets_matched_removal_contrast.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if combined_stability:
        pd.concat(combined_stability, ignore_index=True).to_csv(
            Path(output_dir)
            / "all_datasets_fold_stability_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return results


# -----------------------------------------------------------------------------
# VSCode direct-run configuration
# Edit only this block when changing the formal experiment.
# -----------------------------------------------------------------------------

DATASET_PATHS = [
    r"codeanddata\corel5k_3.mat",
    r"codeanddata\HumanPseAAC_3.mat",
    r"codeanddata\mediamill_3.mat",
    r"codeanddata\YeastBP.mat",
    r"codeanddata\YeastMF.mat",
]
OUTPUT_DIR = r"idpml_v7"

DEFAULT_SELECTOR_CONFIG = SelectorConfig(
    epsilon_mode="feature_gain",
    epsilon_ratio=0.01,
    necessity_mode="exact",
    enable_group_diagnostic=True,
    group_diagnostic_only_when_no_singleton=False,
    group_diagnostic_binary_refine=True,
    random_state=42,
    verbose=True,
)

DEFAULT_EVALUATION_CONFIG = EvaluationConfig(
    svm_c=1.0,
    svm_max_iter=50000,
    svm_n_jobs=1,
    random_state=42,
    fit_full_after_cv=True,
    enable_matched_removal=True,
    matched_removal_budgets=(5, 10, 20),
    matched_removal_random_repeats=10,
    enable_fold_group_diagnostic=True,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        nargs="+",
        default=None,
        help="One or more MAT datasets. Defaults to editable DATASET_PATHS.",
    )
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--epsilon-mode",
        choices=["feature_gain", "total_objective"],
        default=None,
        help=(
            "Tolerance denominator: total feature-attributable gain (v3 "
            "default) or the full objective (v2 compatibility)."
        ),
    )
    parser.add_argument(
        "--epsilon-ratio",
        type=float,
        default=None,
        help="Eta in epsilon=eta*B. The default is 0.01.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "fast", "exact"],
        default=None,
        help="Necessary-feature certification mode.",
    )
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel label jobs for LinearSVC.")
    parser.add_argument(
        "--svm-max-iter",
        type=int,
        default=None,
        help="Maximum iterations for every label-wise LinearSVC fit.",
    )
    parser.add_argument("--no-full-fit", action="store_true", help="Skip final all-sample model fit.")
    parser.add_argument(
        "--group-diagnostic",
        action="store_true",
        help=(
            "Explicitly enable the default cumulative group diagnostic."
        ),
    )
    parser.add_argument(
        "--group-diagnostic-all",
        action="store_true",
        help="Run the group diagnostic even when singleton necessary features exist.",
    )
    parser.add_argument(
        "--no-group-diagnostic",
        action="store_true",
        help="Skip the default final cumulative group diagnostic.",
    )
    parser.add_argument(
        "--no-fold-group-diagnostic",
        action="store_true",
        help=(
            "Do not search for a jointly necessary prefix inside evaluation "
            "training folds that have no singleton necessary feature."
        ),
    )
    parser.add_argument(
        "--no-matched-removal",
        action="store_true",
        help="Skip the held-out matched-removal validation.",
    )
    parser.add_argument(
        "--matched-budgets",
        nargs="+",
        type=int,
        default=None,
        help="Percentage budgets for matched removal. Default: 5 10 20.",
    )
    parser.add_argument(
        "--matched-random-repeats",
        type=int,
        default=None,
        help="Number of matched random-removal controls. Default: 10.",
    )
    parser.add_argument(
        "--group-max-features",
        type=int,
        default=None,
        help="Maximum cumulative group size tested by --group-diagnostic.",
    )
    parser.add_argument(
        "--group-max-fraction",
        type=float,
        default=None,
        help="Maximum original-dimension fraction tested by --group-diagnostic.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce selector logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = DATASET_PATHS if args.data is None else args.data
    selector_config = DEFAULT_SELECTOR_CONFIG
    eval_config = DEFAULT_EVALUATION_CONFIG
    if args.epsilon_mode is not None:
        selector_config = replace(
            selector_config,
            epsilon_mode=args.epsilon_mode,
        )
    if args.epsilon_ratio is not None:
        if not 0.0 < args.epsilon_ratio <= 1.0:
            raise ValueError("--epsilon-ratio must be in (0, 1].")
        selector_config = replace(
            selector_config,
            epsilon_ratio=args.epsilon_ratio,
        )
    if args.mode is not None:
        selector_config = replace(selector_config, necessity_mode=args.mode)
    if args.quiet:
        selector_config = replace(selector_config, verbose=False)
    if args.n_jobs is not None:
        eval_config = replace(eval_config, svm_n_jobs=args.n_jobs)
    if args.svm_max_iter is not None:
        eval_config = replace(eval_config, svm_max_iter=args.svm_max_iter)
    if args.no_full_fit:
        eval_config = replace(eval_config, fit_full_after_cv=False)
    if args.no_fold_group_diagnostic:
        eval_config = replace(
            eval_config,
            enable_fold_group_diagnostic=False,
        )
    if args.no_matched_removal:
        eval_config = replace(
            eval_config,
            enable_matched_removal=False,
        )
    if args.matched_budgets is not None:
        if (
            len(args.matched_budgets) == 0
            or any(
                budget < 1 or budget > 100
                for budget in args.matched_budgets
            )
        ):
            raise ValueError("--matched-budgets values must be in [1, 100].")
        eval_config = replace(
            eval_config,
            matched_removal_budgets=tuple(
                dict.fromkeys(args.matched_budgets)
            ),
        )
    if args.matched_random_repeats is not None:
        if args.matched_random_repeats < 1:
            raise ValueError("--matched-random-repeats must be at least 1.")
        eval_config = replace(
            eval_config,
            matched_removal_random_repeats=(
                args.matched_random_repeats
            ),
        )
    if args.no_group_diagnostic:
        selector_config = replace(
            selector_config,
            enable_group_diagnostic=False,
        )
    elif args.group_diagnostic or args.group_diagnostic_all:
        selector_config = replace(
            selector_config,
            enable_group_diagnostic=True,
            group_diagnostic_only_when_no_singleton=False,
        )
    if args.group_max_features is not None:
        selector_config = replace(
            selector_config,
            group_diagnostic_max_features=max(1, args.group_max_features),
        )
    if args.group_max_fraction is not None:
        if not 0.0 < args.group_max_fraction <= 1.0:
            raise ValueError("--group-max-fraction must be in (0, 1].")
        selector_config = replace(
            selector_config,
            group_diagnostic_max_fraction=args.group_max_fraction,
        )
    run_multiple_datasets(paths, args.output, selector_config, eval_config)


if __name__ == "__main__":
    main()
