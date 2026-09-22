"""PML feature necessity v4.4: collective-search alignment and result reuse.

This file is intentionally kept separate from the v4.3 implementation.  It has
two execution modes configured in the block at the bottom:

1. ``reuse_existing`` (default): audit an existing v4.3 result directory and
   retain only collective results that already satisfy the v4.4 definition.
   No selector, SVM, or ordinary Top-k experiment is rerun.
2. ``full_experiment``: run new datasets through the v4.3 pipeline with the
   corrected policy: collective search is a fallback only when no singleton is
   necessary, and it is attempted only after every singleton has a certified
   decision.  In that case all features are certified removable, so the
   weight-ranked prefixes are exactly prefixes of the removable universe.

The base implementation path is editable, allowing the experimental-machine
copy of v4.3 to be used without renaming it as a Python module.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import List, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# USER CONFIGURATION: edit this block in VSCode, then use "Run Python File".
# =============================================================================

RUN_MODE = "reuse_existing"  # "reuse_existing" or "full_experiment"

# Point this to the latest v4.3 program on the experimental machine.
BASE_IMPLEMENTATION_PATH = r"partial_id_pml_fs_v4.py"

# Used by reuse_existing.  Conventional results remain in this directory.
SOURCE_RESULTS_DIR = r"idpml_v8"
COLLECTIVE_OUTPUT_DIR = r"idpml_v9_collective"

# Empty means: process every complete dataset found in SOURCE_RESULTS_DIR.
# Otherwise enter result stems exactly as they appear before
# "_feature_details.csv", for example ["YeastMF", "enron_5"].
DATASET_NAMES: Sequence[str] = ()

# Used only by full_experiment for future datasets.
DATASET_PATHS: Sequence[str] = (
    r"codeanddata\corel5k_3.mat",
    r"codeanddata\HumanPseAAC_3.mat",
    r"codeanddata\mediamill_3.mat",
    r"codeanddata\YeastBP.mat",
    r"codeanddata\YeastMF.mat",
)
FULL_EXPERIMENT_OUTPUT_DIR = r"idpml_v9_full"


# =============================================================================
# Loading and shared checks
# =============================================================================


def load_base_implementation(filepath: str) -> ModuleType:
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Base v4.3 implementation not found: {path}. "
            "Edit BASE_IMPLEMENTATION_PATH in the configuration block."
        )
    module_name = "partial_id_pml_fs_v43_core"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import the base implementation from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


CORE = None
_ORIGINAL_GROUP_DIAGNOSTIC = None

METRIC_COLUMNS = [
    "RankingLoss",
    "HammingLoss",
    "CoverageError",
    "OneError",
    "AveragePrecision",
    "MicroF1",
    "MacroF1",
]


def _as_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="raise").astype(np.int64)


def _discover_dataset_names(source: Path) -> List[str]:
    suffix = "_feature_details.csv"
    names = sorted(
        path.name[: -len(suffix)]
        for path in source.glob(f"*{suffix}")
    )
    if not names:
        raise FileNotFoundError(
            f"No '*{suffix}' files were found in {source.resolve()}."
        )
    return names


def _annotate_collective_frame(
    frame: pd.DataFrame,
    candidate_universe_size: int,
    reuse_status: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["CandidateUniverse"] = "CertifiedRemovable"
    out["CandidateUniverseSize"] = int(candidate_universe_size)
    out["AllMembersSingletonRemovable"] = True
    out["CollectivePolicy"] = "FallbackWhenNoSingletonNecessary"
    out["ReuseStatus"] = reuse_status
    return out


def _write_if_nonempty(frame: pd.DataFrame, path: Path) -> None:
    if not frame.empty:
        frame.to_csv(path, index=False, encoding="utf-8-sig")


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
    rows = []
    for keys, subset in raw_df.groupby(group_columns, sort=False, dropna=False):
        row = dict(zip(group_columns, keys))
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
    if raw_df.empty:
        return pd.DataFrame()
    rows = []
    keys = ["Dataset", "RemovalTarget", "BudgetPercent", "NumFeatures"]
    for values, subset in raw_df.groupby(keys, sort=False, dropna=False):
        target = subset[subset["Variant"] == "TargetRemoved"]
        random = subset[subset["Variant"] == "RandomRemoved"]
        if target.empty or random.empty:
            continue
        row = dict(zip(keys, values))
        fold_records = []
        for fold in sorted(set(target["Fold"]) & set(random["Fold"])):
            target_fold = target[target["Fold"] == fold].iloc[0]
            random_fold = random[random["Fold"] == fold]
            record = {"Fold": float(fold)}
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


# =============================================================================
# Existing-result reuse
# =============================================================================


def reuse_existing_results() -> None:
    """Create manuscript-aligned collective files without rerunning experiments.

    A v4.3 group result is exactly reusable when NumNecessary == 0 and
    NumUnresolved == 0.  Under this condition every feature is certified
    removable, and v4.3's norm-ranked full prefix is identical to v4.4's prefix
    over the certified-removable universe.  Results with singleton necessity
    are skipped because collective search is now a complementary fallback.
    Any unresolved singleton makes reuse unsafe and is reported as requiring a
    targeted refit rather than being silently accepted.
    """

    source = Path(SOURCE_RESULTS_DIR)
    output = Path(COLLECTIVE_OUTPUT_DIR)
    if not source.exists():
        raise FileNotFoundError(f"Source result directory not found: {source.resolve()}")
    output.mkdir(parents=True, exist_ok=True)

    names = list(DATASET_NAMES) if DATASET_NAMES else _discover_dataset_names(source)
    audit_rows = []
    all_matched_raw = []

    for name in names:
        detail_path = source / f"{name}_feature_details.csv"
        if not detail_path.exists():
            raise FileNotFoundError(f"Missing required file: {detail_path.resolve()}")

        detail = pd.read_csv(detail_path)
        necessary_count = int(_as_int(detail["IsNecessary"]).sum())
        unresolved_count = int(_as_int(detail["IsUnresolved"]).sum())
        dimension = int(len(detail))

        if necessary_count > 0:
            full_status = "SkippedSingletonNecessary"
        elif unresolved_count > 0:
            full_status = "NeedsTargetedRefitUnresolved"
        else:
            old_group_path = source / f"{name}_group_diagnostic.csv"
            if old_group_path.exists():
                group = pd.read_csv(old_group_path)
                group = _annotate_collective_frame(
                    group,
                    candidate_universe_size=dimension,
                    reuse_status="ExactlyReusableV43",
                )
                group.to_csv(
                    output / f"{name}_group_diagnostic.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                full_status = "Reused"
            else:
                full_status = "MissingGroupDiagnostic"

        audit_rows.append(
            {
                "Dataset": name,
                "Level": "FullData",
                "Fold": np.nan,
                "NumNecessary": necessary_count,
                "NumUnresolved": unresolved_count,
                "CandidateUniverseSize": (
                    dimension
                    if necessary_count == 0 and unresolved_count == 0
                    else 0
                ),
                "Action": full_status,
            }
        )

        fold_sets_path = source / f"{name}_fold_feature_sets.csv"
        if not fold_sets_path.exists():
            audit_rows.append(
                {
                    "Dataset": name,
                    "Level": "Fold",
                    "Fold": np.nan,
                    "NumNecessary": np.nan,
                    "NumUnresolved": np.nan,
                    "CandidateUniverseSize": np.nan,
                    "Action": "MissingFoldFeatureSets",
                }
            )
            continue

        fold_sets = pd.read_csv(fold_sets_path)
        fold_sets["Fold"] = _as_int(fold_sets["Fold"])
        fold_sets["NumNecessary"] = _as_int(fold_sets["NumNecessary"])
        fold_sets["NumUnresolved"] = _as_int(fold_sets["NumUnresolved"])
        reusable_mask = (
            (fold_sets["NumNecessary"] == 0)
            & (fold_sets["NumUnresolved"] == 0)
        )
        reusable_folds = set(fold_sets.loc[reusable_mask, "Fold"].tolist())

        for row in fold_sets.itertuples(index=False):
            if int(row.NumNecessary) > 0:
                action = "SkippedSingletonNecessary"
            elif int(row.NumUnresolved) > 0:
                action = "NeedsTargetedRefitUnresolved"
            else:
                action = "Reusable"
            audit_rows.append(
                {
                    "Dataset": name,
                    "Level": "Fold",
                    "Fold": int(row.Fold),
                    "NumNecessary": int(row.NumNecessary),
                    "NumUnresolved": int(row.NumUnresolved),
                    "CandidateUniverseSize": (
                        dimension if action == "Reusable" else 0
                    ),
                    "Action": action,
                }
            )

        old_fold_group_path = source / f"{name}_fold_group_diagnostic.csv"
        if reusable_folds and old_fold_group_path.exists():
            fold_group = pd.read_csv(old_fold_group_path)
            fold_group["Fold"] = _as_int(fold_group["Fold"])
            fold_group = fold_group[fold_group["Fold"].isin(reusable_folds)]
            fold_group = _annotate_collective_frame(
                fold_group,
                candidate_universe_size=dimension,
                reuse_status="ExactlyReusableV43",
            )
            _write_if_nonempty(
                fold_group,
                output / f"{name}_fold_group_diagnostic.csv",
            )

        old_matched_path = source / f"{name}_matched_removal_raw.csv"
        if reusable_folds and old_matched_path.exists():
            matched = pd.read_csv(old_matched_path)
            matched["Fold"] = _as_int(matched["Fold"])
            matched = matched[
                matched["Fold"].isin(reusable_folds)
                & (matched["RemovalTarget"] == "JointGroup")
            ].copy()
            if not matched.empty:
                matched["CollectivePolicy"] = (
                    "FallbackWhenNoSingletonNecessary"
                )
                matched["ReuseStatus"] = "ExactlyReusableV43"
                matched.to_csv(
                    output / f"{name}_matched_removal_raw.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                summarize_matched_removal(matched).to_csv(
                    output / f"{name}_matched_removal_summary.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                matched_removal_contrast_frame(matched).to_csv(
                    output / f"{name}_matched_removal_contrast.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                all_matched_raw.append(matched)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(
        output / "collective_reuse_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if all_matched_raw:
        combined = pd.concat(all_matched_raw, ignore_index=True)
        summarize_matched_removal(combined).to_csv(
            output / "all_datasets_matched_removal_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        matched_removal_contrast_frame(combined).to_csv(
            output / "all_datasets_matched_removal_contrast.csv",
            index=False,
            encoding="utf-8-sig",
        )

    counts = audit["Action"].value_counts().to_dict()
    print(f"[v4.4] manuscript-aligned files saved to {output.resolve()}")
    print(f"[v4.4] audit counts: {counts}")
    if (audit["Action"] == "NeedsTargetedRefitUnresolved").any():
        print(
            "[v4.4] WARNING: unresolved cases were not reused. Run a targeted "
            "fresh experiment for only those datasets before reporting them."
        )


# =============================================================================
# Corrected fresh run for future datasets
# =============================================================================


def _empty_group_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "FirstCertifiedNecessaryPrefix",
            "ElapsedSeconds",
        ]
    )


def _v44_group_diagnostic(X, Y, selector, base_fit, singleton_result, ranking):
    """Enforce the v4.4 collective-search eligibility conditions.

    The v4.3 outer pipeline reconstructs a discovered group from its global
    ranking.  To preserve that invariant without rewriting the full CV driver,
    v4.4 runs group search only after a complete singleton partition establishes
    that every feature is removable.  Then the global norm ranking and the
    certified-removable ranking are identical.
    """

    if np.any(singleton_result.necessary):
        return _empty_group_frame()
    if np.any(singleton_result.unresolved):
        if selector.config.verbose:
            print(
                "[GroupDiagnostic] skipped: collective membership requires "
                "certified singleton removability, but unresolved features remain."
            )
        return _empty_group_frame()

    importance = np.linalg.norm(base_fit.W, axis=1)
    removable_ranking = np.argsort(-importance, kind="stable").astype(np.int64)
    if _ORIGINAL_GROUP_DIAGNOSTIC is None:
        raise RuntimeError("The base v4.3 group diagnostic has not been loaded.")
    result = _ORIGINAL_GROUP_DIAGNOSTIC(
        X,
        Y,
        selector,
        base_fit,
        singleton_result,
        removable_ranking,
    )
    if not result.empty:
        result = _annotate_collective_frame(
            result,
            candidate_universe_size=X.shape[1],
            reuse_status="FreshV44",
        )
    return result


def run_full_experiment() -> None:
    """Run future datasets with v4.4's corrected collective-search policy."""

    global CORE, _ORIGINAL_GROUP_DIAGNOSTIC
    CORE = load_base_implementation(BASE_IMPLEMENTATION_PATH)
    _ORIGINAL_GROUP_DIAGNOSTIC = CORE.cumulative_group_diagnostic
    CORE.cumulative_group_diagnostic = _v44_group_diagnostic
    selector_config = replace(
        CORE.DEFAULT_SELECTOR_CONFIG,
        necessity_mode="exact",
        enable_group_diagnostic=True,
        group_diagnostic_only_when_no_singleton=True,
        group_diagnostic_binary_refine=True,
    )
    evaluation_config = replace(
        CORE.DEFAULT_EVALUATION_CONFIG,
        svm_n_jobs=1,
        svm_max_iter=50000,
        fit_full_after_cv=True,
        enable_fold_group_diagnostic=True,
        enable_matched_removal=True,
    )
    CORE.run_multiple_datasets(
        list(DATASET_PATHS),
        FULL_EXPERIMENT_OUTPUT_DIR,
        selector_config,
        evaluation_config,
    )


def main() -> None:
    mode = RUN_MODE.strip().lower()
    if mode == "reuse_existing":
        reuse_existing_results()
    elif mode == "full_experiment":
        run_full_experiment()
    else:
        raise ValueError(
            "RUN_MODE must be 'reuse_existing' or 'full_experiment', "
            f"got {RUN_MODE!r}."
        )


if __name__ == "__main__":
    main()
