# FIND: supplementary code and synthetic experiment

This anonymous package accompanies the paper *Beyond a Single Sparse Solution: Certifying Feature Necessity under Inexact Multi-Label Supervision*. It provides the feature-necessity implementation and the controlled synthetic experiment. The benchmark `.mat` files and implementations of external baselines are not included in this code package.

## Files and dependencies

Place the following three Python files in the same directory:

| File | Role |
| --- | --- |
| `partial_id_pml_fs_v4.py` | Core ambiguity-aware objective, optimization, certificates, feature ranking, and evaluation pipeline. |
| `find.py` | FIND's final collective-search policy and benchmark experiment wrapper. It imports the core file above. |
| `synthetic_necessity_experiment.py` | Generates controlled data and evaluates individual and collective certificates. It also imports the core file above. |

Use Python 3.10 or newer with NumPy, pandas, SciPy, and scikit-learn. Matplotlib is needed only to generate the optional synthetic figures. A possible installation command is:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib
```

Both experiment scripts have an editable configuration block near the top. Their `BASE_IMPLEMENTATION_PATH` must point to `partial_id_pml_fs_v4.py`; the default relative filename works when all three files are together.

## Input data

Each benchmark dataset is a MATLAB `.mat` file with three arrays:

- `data`: samples by features;
- `partial_labels`: observed candidate labels;
- `target`: true labels, used for validation and held-out evaluation rather than as the selector's training labels.

The label arrays may be stored as samples by labels or labels by samples; the core loader checks their orientation. For partial multi-label learning, every positive true label must be included in the corresponding candidate set. Use the dataset preparation and evaluation protocol described in the paper when comparing with its benchmark tables.

## Run the controlled synthetic experiment

From the directory containing the three scripts, run:

```bash
python synthetic_necessity_experiment.py
```

The default configuration uses 30 seeds for each of two scenarios and three distractor-label levels. It writes to `synthetic_necessity_results_v2/`. The main outputs are `synthetic_paper_table.csv` (compact paper values), `synthetic_summary.csv` (aggregated results), `synthetic_run_results.csv` (per-run outcomes), `synthetic_feature_results.csv` (singleton certificates), and `synthetic_group_results.csv` (group certificates). If `synthetic_errors.csv` is produced, at least one run failed and should not be silently omitted. The generator can also save the generated `.mat` datasets and a figure; these outputs are optional for rerunning the experiment because the seeds and generation procedure are in the script.

The individual scenario contains one uniquely informative feature. The collective scenario contains two interchangeable features that are individually removable but jointly non-removable. Other features are noise. The candidate sets contain the two true labels plus 0, 2, or 4 distractors.

## Run FIND on benchmark data

`find.py` defaults to `RUN_MODE = "reuse_existing"`. This mode audits and reuses an existing result directory; it requires the configured `SOURCE_RESULTS_DIR` and is **not** a fresh benchmark run. To run new datasets, change `RUN_MODE` to `"full_experiment"`, set `DATASET_PATHS` to the `.mat` files available locally, and select an output directory in its configuration block. Then run:

```bash
python find.py
```

The wrapper applies the paper's collective-search eligibility rule: group search is attempted only after a complete singleton partition certifies that every singleton is removable. The core file's own command-line entry point remains available for inspecting and running its base pipeline, but its historical default dataset list and output name are not the final FIND benchmark configuration.

## Anonymous-distribution note

The scripts contain no author names or institution-specific paths. The synthetic script writes a `synthetic_manifest.json` containing the absolute path of the core file on the machine that ran it. This manifest is not needed to reproduce the reported results; remove its `base_implementation` field or omit the manifest when distributing generated results anonymously. Before uploading any output directory, also inspect filenames and file contents for local paths or identifying metadata.
