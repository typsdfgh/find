"""V4.4-aligned controlled experiment for feature necessity.

Run directly in VSCode.  The individual scenario contains one unique signal;
the collective scenario contains two interchangeable copies and invokes group
search only after every singleton has been certified removable.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# USER CONFIGURATION ---------------------------------------------------------
# V4.4 is an outer policy wrapper around this core implementation.  The same
# eligibility policy is applied explicitly in run_one_condition below.
BASE_IMPLEMENTATION_PATH = r"partial_id_pml_fs_v4.py"
OUTPUT_DIR = r"synthetic_necessity_results_v2"
SCENARIOS: Sequence[str] = ("individual", "collective")
NUM_SEEDS = 30
FIRST_SEED = 100
NUM_DISTRACTOR_LABELS: Sequence[int] = (0, 2, 4)
N_SAMPLES, N_FEATURES, N_LABELS = 2000, 20, 8
SIGNAL_STRENGTH, SIGNAL_NOISE_STD = 2.0, 0.20
ETA, LAMBDA_RATIO, GAMMA_RATIO = 0.01, 1e-3, 1e-4
SAVE_MAT_DATASETS, MAKE_FIGURES, FAIL_FAST = True, True, False
MAX_ITER, OFF_MAX_ITER = 300, 300
ADAPTIVE_REFINEMENT_ROUNDS = 5
ADAPTIVE_REFINEMENT_MAX_ITER = 2000
CERTIFICATE_GAP_RATIO_TARGET = 0.02

UNIQUE_FEATURE = 0
REDUNDANT_GROUP = np.asarray([0, 1], dtype=np.int64)
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else SCRIPT_DIR / path


def load_core(filepath: str) -> ModuleType:
    path = resolve_path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Base implementation not found: {path}")
    spec = importlib.util.spec_from_file_location("synthetic_necessity_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def role(scenario: str, index: int) -> str:
    if scenario == "individual" and index == UNIQUE_FEATURE:
        return "UniqueSignal"
    if scenario == "collective" and index in REDUNDANT_GROUP:
        return "RedundantMember"
    return "Noise"


def expected_status(scenario: str, index: int) -> str:
    return "Necessary" if scenario == "individual" and index == UNIQUE_FEATURE else "Removable"


def status(necessary: bool, unresolved: bool) -> str:
    return "Necessary" if necessary else ("Unresolved" if unresolved else "Removable")


def generate_data(scenario: str, seed: int, distractors: int):
    if scenario not in SCENARIOS or N_FEATURES < 4 or N_LABELS != 8:
        raise ValueError("Invalid scenario or dimensions.")
    if not 0 <= distractors <= N_LABELS - 2:
        raise ValueError("Invalid distractor count.")
    rng = np.random.default_rng(seed)
    regimes = np.tile(np.arange(2), int(math.ceil(N_SAMPLES / 2)))[:N_SAMPLES]
    rng.shuffle(regimes)
    latent = np.where(regimes == 0, SIGNAL_STRENGTH, -SIGNAL_STRENGTH)
    latent += SIGNAL_NOISE_STD * rng.normal(size=N_SAMPLES)
    X = rng.normal(size=(N_SAMPLES, N_FEATURES)).astype(np.float32)
    if scenario == "individual":
        X[:, UNIQUE_FEATURE] = latent
    else:
        X[:, REDUNDANT_GROUP[0]] = latent
        X[:, REDUNDANT_GROUP[1]] = latent
    target = np.zeros((N_SAMPLES, N_LABELS), dtype=np.int8)
    target[regimes == 0, 0:2] = 1
    target[regimes == 1, 2:4] = 1
    partial = target.copy()
    for i in range(N_SAMPLES):
        if distractors:
            candidates = np.flatnonzero(target[i] == 0)
            partial[i, rng.choice(candidates, distractors, replace=False)] = 1
    assert np.all(target.sum(1) == 2)
    assert np.all(partial.sum(1) == 2 + distractors)
    return X, partial, target, regimes.astype(np.int8)


def standardize(X: np.ndarray) -> np.ndarray:
    mean, scale = X.mean(0, dtype=np.float64), X.std(0, dtype=np.float64)
    return ((X - mean) / np.where(scale > 1e-12, scale, 1.0)).astype(np.float32)


def save_mat(path: Path, X, partial, target, regimes, scenario, distractors):
    from scipy.io import savemat
    codes = {"Noise": 0, "UniqueSignal": 1, "RedundantMember": 2}
    savemat(path, {
        "data": X.astype(np.float32),
        "partial_labels": partial.T.astype(np.int8),
        "target": target.T.astype(np.int8),
        "sample_regime": regimes.reshape(-1, 1),
        "feature_role_code": np.asarray(
            [codes[role(scenario, i)] for i in range(N_FEATURES)], dtype=np.int8
        ).reshape(-1, 1),
        "scenario": np.asarray([scenario], dtype=object),
        "num_distractor_labels": np.asarray([[distractors]], dtype=np.int16),
    }, do_compression=True)


def selector_config(core, seed):
    return core.SelectorConfig(
        epsilon_mode="feature_gain", epsilon_ratio=ETA, necessity_mode="exact",
        max_iter=MAX_ITER, off_max_iter=OFF_MAX_ITER, tol=1e-7,
        certificate_refine_iterations=60,
        adaptive_refinement_rounds=ADAPTIVE_REFINEMENT_ROUNDS,
        adaptive_refinement_max_iter=ADAPTIVE_REFINEMENT_MAX_ITER,
        adaptive_min_tol=1e-10,
        certificate_gap_ratio_target=CERTIFICATE_GAP_RATIO_TARGET,
        enable_group_diagnostic=True,
        group_diagnostic_only_when_no_singleton=True,
        group_diagnostic_binary_refine=True,
        verbose=False, random_state=seed,
    )


def annotate_features(frame, scenario, seed, distractors):
    frame = frame.copy()
    frame.insert(0, "Scenario", scenario)
    frame.insert(1, "Seed", seed)
    frame.insert(2, "NumDistractorLabels", distractors)
    frame.insert(3, "CandidateSetSize", 2 + distractors)
    frame.insert(4, "Eta", ETA)
    # feature_detail_frame is rank-sorted: always join by FeatureIndex0.
    frame["Role"] = [role(scenario, int(i)) for i in frame.FeatureIndex0]
    frame["Status"] = [status(bool(n), bool(u)) for n, u in zip(
        frame.IsNecessary, frame.IsUnresolved
    )]
    frame["ExpectedStatus"] = [
        expected_status(scenario, int(i)) for i in frame.FeatureIndex0
    ]
    frame["CorrectSingletonDecision"] = frame.Status.eq(frame.ExpectedStatus)
    frame["RhoLower/Eta"] = frame.RhoLower / max(ETA, 1e-12)
    frame["RhoUpper/Eta"] = frame.RhoUpper / max(ETA, 1e-12)
    return frame


def parse_indices(value) -> np.ndarray:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return np.asarray([], dtype=np.int64)
    return np.asarray([int(x) for x in text.split(";")], dtype=np.int64)


def run_one_condition(core, scenario, seed, distractors, dataset_dir):
    X_raw, partial, target, regimes = generate_data(scenario, seed, distractors)
    if SAVE_MAT_DATASETS:
        save_mat(
            dataset_dir / f"synthetic_{scenario}_d{distractors}_seed{seed:04d}.mat",
            X_raw, partial, target, regimes, scenario, distractors,
        )
    X = standardize(X_raw)
    config = selector_config(core, seed)
    started = time.perf_counter()
    selector, fit, epsilon_reference, *_ = core._fit_selector_for_fold(
        X, partial, LAMBDA_RATIO, GAMMA_RATIO, config
    )
    necessity = core.certify_necessary_features(
        X, partial, selector, fit, config, epsilon_reference
    )
    features = annotate_features(
        core.feature_detail_frame(fit, necessity), scenario, seed, distractors
    )
    by_id = features.set_index("FeatureIndex0", drop=False)
    n_necessary = int(features.Status.eq("Necessary").sum())
    n_unresolved = int(features.Status.eq("Unresolved").sum())
    eligible = n_necessary == 0 and n_unresolved == 0
    groups, discovered = pd.DataFrame(), np.asarray([], dtype=np.int64)
    collective_recovery = np.nan

    # Exact v4.4 policy and automatic (not oracle-given) prefix discovery.
    if scenario == "collective" and eligible:
        ranking = np.argsort(-np.linalg.norm(fit.W, axis=1), kind="stable")
        groups = core.cumulative_group_diagnostic(
            X, partial, selector, fit, necessity, ranking
        )
        if not groups.empty:
            groups.insert(0, "Scenario", scenario)
            groups.insert(1, "Seed", seed)
            groups.insert(2, "NumDistractorLabels", distractors)
            groups.insert(3, "CandidateSetSize", 2 + distractors)
            endpoint = groups[groups.FirstCertifiedNecessaryPrefix.astype(bool)]
            if not endpoint.empty:
                discovered = parse_indices(endpoint.iloc[0].FeatureIndices0)
        collective_recovery = float(
            discovered.size > 0
            and set(REDUNDANT_GROUP).issubset(set(discovered))
        )

    if scenario == "individual":
        individual_recovery = float(by_id.loc[UNIQUE_FEATURE].Status == "Necessary")
        noise_ids = [i for i in range(N_FEATURES) if i != UNIQUE_FEATURE]
    else:
        individual_recovery = np.nan
        noise_ids = [i for i in range(N_FEATURES) if i not in REDUNDANT_GROUP]
    noise_false = float(by_id.loc[noise_ids].Status.eq("Necessary").mean())
    full = float(
        features.CorrectSingletonDecision.all() and n_unresolved == 0
        and (scenario == "individual" or collective_recovery == 1.0)
    )
    row: Dict[str, object] = {
        "Scenario": scenario, "Seed": seed,
        "NumDistractorLabels": distractors,
        "CandidateSetSize": 2 + distractors, "Eta": ETA,
        "NumCertifiedNecessary": n_necessary,
        "NumCertifiedRemovable": int(features.Status.eq("Removable").sum()),
        "NumUnresolved": n_unresolved,
        "CollectiveSearchEligible": float(eligible),
        "CollectiveSearchExecuted": float(scenario == "collective" and eligible),
        "IndividualRecovery": individual_recovery,
        "CollectiveRecovery": collective_recovery,
        "SingletonDecisionAccuracy": float(features.CorrectSingletonDecision.mean()),
        "NoiseFalseNecessaryRate": noise_false,
        "FullPatternRecovery": full,
        "DiscoveredGroupSize": int(discovered.size),
        "DiscoveredFeatureIndices0": ";".join(map(str, discovered)),
        "BaseCertificateConverged": bool(fit.certificate_converged),
        "InterceptCertificateConverged": bool(necessity.intercept_certificate_converged),
        "AllExactRefitsConverged": bool(
            necessity.exact_refits == necessity.exact_refits_converged
        ),
        "ExactRefits": int(necessity.exact_refits),
        "TotalRunSeconds": float(time.perf_counter() - started),
    }
    return features, groups, row


def summarize(run_frame):
    metrics = [
        "CollectiveSearchEligible", "CollectiveSearchExecuted",
        "IndividualRecovery", "CollectiveRecovery", "SingletonDecisionAccuracy",
        "NoiseFalseNecessaryRate", "FullPatternRecovery", "TotalRunSeconds",
    ]
    rows: List[Dict[str, object]] = []
    for (scenario, distractors), part in run_frame.groupby(
        ["Scenario", "NumDistractorLabels"], sort=True
    ):
        row = {
            "Scenario": scenario, "NumDistractorLabels": int(distractors),
            "CandidateSetSize": int(part.CandidateSetSize.iloc[0]),
            "NumRuns": int(len(part)), "Eta": ETA,
        }
        for metric in metrics:
            values = part[metric].dropna().astype(float).to_numpy()
            if not values.size:
                mean = sd = lo = hi = np.nan
            else:
                mean = float(values.mean())
                sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
                half = 1.96 * sd / math.sqrt(values.size)
                lo, hi = mean - half, mean + half
                if metric != "TotalRunSeconds":
                    lo, hi = max(0.0, lo), min(1.0, hi)
            row.update({f"Mean{metric}": mean, f"Std{metric}": sd,
                        f"CI95Lower{metric}": lo, f"CI95Upper{metric}": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def paper_table(summary):
    pct = lambda name: (100 * summary[name]).round(1)
    return pd.DataFrame({
        "Scenario": summary.Scenario,
        "Distractors": summary.NumDistractorLabels.astype(int),
        "CandidateSize": summary.CandidateSetSize.astype(int),
        "IndividualRecovery(%)": pct("MeanIndividualRecovery"),
        "CollectiveRecovery(%)": pct("MeanCollectiveRecovery"),
        "SingletonAccuracy(%)": pct("MeanSingletonDecisionAccuracy"),
        "NoiseFalseCertificate(%)": pct("MeanNoiseFalseNecessaryRate"),
        "FullPatternRecovery(%)": pct("MeanFullPatternRecovery"),
    })


def make_figure(summary, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib unavailable; CSV outputs were written.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), constrained_layout=True)
    for ax, scenario, metric in zip(
        axes, ("individual", "collective"),
        ("MeanIndividualRecovery", "MeanCollectiveRecovery"),
    ):
        part = summary[summary.Scenario == scenario]
        ax.plot(part.CandidateSetSize, part[metric], "o-", label="Target recovery")
        ax.plot(part.CandidateSetSize, 1 - part.MeanNoiseFalseNecessaryRate,
                "s--", label="Noise rejection")
        ax.plot(part.CandidateSetSize, part.MeanFullPatternRecovery,
                "^-", label="Full pattern")
        ax.set(title=scenario.capitalize(), xlabel="Candidate-set size", ylim=(-0.03, 1.05))
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Recovery rate")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(output_dir / "synthetic_structure_figure.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "synthetic_structure_figure.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    output_dir = resolve_path(OUTPUT_DIR)
    dataset_dir = output_dir / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    base_path = resolve_path(BASE_IMPLEMENTATION_PATH).resolve()
    core = load_core(str(base_path))
    features_all, groups_all, rows, errors = [], [], [], []
    total = len(SCENARIOS) * len(NUM_DISTRACTOR_LABELS) * NUM_SEEDS
    counter, started = 0, time.perf_counter()
    for scenario in SCENARIOS:
        for distractors in NUM_DISTRACTOR_LABELS:
            for seed in range(FIRST_SEED, FIRST_SEED + NUM_SEEDS):
                counter += 1
                print(f"[{counter:03d}/{total}] {scenario}, distractors={distractors}, seed={seed}")
                try:
                    features, groups, row = run_one_condition(
                        core, scenario, seed, int(distractors), dataset_dir
                    )
                    features_all.append(features)
                    if not groups.empty:
                        groups_all.append(groups)
                    rows.append(row)
                    print(f"    necessary={row['NumCertifiedNecessary']}, "
                          f"unresolved={row['NumUnresolved']}, full={int(row['FullPatternRecovery'])}")
                except Exception as exc:
                    errors.append({
                        "Scenario": scenario, "Seed": seed,
                        "NumDistractorLabels": int(distractors),
                        "ErrorType": type(exc).__name__, "ErrorMessage": str(exc),
                        "Traceback": traceback.format_exc(),
                    })
                    print(f"    ERROR: {type(exc).__name__}: {exc}")
                    if FAIL_FAST:
                        raise
    if not rows:
        raise RuntimeError("No condition completed successfully.")
    feature_frame = pd.concat(features_all, ignore_index=True)
    group_frame = pd.concat(groups_all, ignore_index=True) if groups_all else pd.DataFrame()
    run_frame = pd.DataFrame(rows).sort_values(
        ["Scenario", "NumDistractorLabels", "Seed"], kind="stable"
    )
    summary = summarize(run_frame)
    ready = paper_table(summary)
    for filename, frame in {
        "synthetic_feature_results.csv": feature_frame,
        "synthetic_group_results.csv": group_frame,
        "synthetic_run_results.csv": run_frame,
        "synthetic_summary.csv": summary,
        "synthetic_paper_table.csv": ready,
    }.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    if errors:
        pd.DataFrame(errors).to_csv(
            output_dir / "synthetic_errors.csv", index=False, encoding="utf-8-sig"
        )
    if MAKE_FIGURES:
        make_figure(summary, output_dir)
    manifest = {
        "experiment": "v44_aligned_individual_and_collective_necessity",
        "description": "Separate end-to-end individual and collective scenarios.",
        "base_implementation": str(base_path),
        "base_implementation_sha256": sha256(base_path),
        "python": sys.version, "platform": platform.platform(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "configuration": {
            "scenarios": list(SCENARIOS), "num_seeds": NUM_SEEDS,
            "first_seed": FIRST_SEED,
            "num_distractor_labels": list(map(int, NUM_DISTRACTOR_LABELS)),
            "n_samples": N_SAMPLES, "n_features": N_FEATURES,
            "n_labels": N_LABELS, "eta": ETA,
            "lambda_ratio": LAMBDA_RATIO, "gamma_ratio": GAMMA_RATIO,
        },
        "v44_policy": {
            "collective_only_if_no_necessary": True,
            "collective_only_if_no_unresolved": True,
            "automatic_prefix_search": True,
        },
        "completed_runs": len(run_frame), "failed_runs": len(errors),
        "wall_seconds": time.perf_counter() - started,
    }
    with (output_dir / "synthetic_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("\nPaper-ready summary:")
    print(ready.to_string(index=False))
    print(f"\nResults written to: {output_dir.resolve()}")
    if errors:
        raise RuntimeError(f"{len(errors)} conditions failed; inspect synthetic_errors.csv")


if __name__ == "__main__":
    main()
