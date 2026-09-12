# AI Dependency and Cognitive Offloading Among University Students

An explainable machine-learning portfolio study using self-reported survey data.
**This is exploratory:** primary Cronbach's alpha values are weak, duplicate
origins are unknown, and no cognitive ability or clinical condition is measured.

## Start here

- [Anaconda-compatible notebook](AI_Dependency_Study_anaconda_compatible.ipynb): sequential objectives, executable code, actual outputs, and interpretation.
- [Final research report](FINAL_REPORT.md): methods, results, seven research answers, and limitations.
- [HTML report and figure gallery](FINAL_REPORT.html): open locally in a browser; keep the outputs/figures folder beside it.
- [Model comparison](outputs/tables/model_comparison.csv).
- [Reliability and duplicate sensitivity](outputs/tables/reliability.csv).

## Actual results

- Raw dataset: 2,613 rows, 30 columns; no missing responses.
- Primary analysis: 2,162 distinct complete response patterns; 451 repetitions beyond first copies.
- CV-selected model: B / Gradient Boosting.
- Held-out MAE 0.3839; RMSE 0.5036; R² 0.2923.
- Selected K-Means k=3, silhouette=0.2234; partitions do not establish natural student categories.
- SHAP 0.52.0 executed on 250 held-out patterns; additivity verified.

## Reproduce

Use Python 3.12 from the repository root. Place the original CSV there with the
exact filename below. The data is not downloaded or fabricated by the project.

```text
_(AI_Cognitive_Dependence_Survey_Data_BD_StudentsResponses) - Sheet1.csv
```

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python build_project.py
```

SHAP is optional: install `requirements-optional.txt` to attempt tree SHAP.
If SHAP cannot be imported or its additivity check fails, the exception is reported
and the already-executed permutation analysis is used. XGBoost is not required.
The core notebook also runs sequentially in an IDE with the installed Python kernel.
For a browser notebook interface, install JupyterLab separately if desired.

## Local Streamlit dashboard

After running all notebook cells, start the dashboard with:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Use the `python -m streamlit` form because it works even when Streamlit's Scripts
folder is not on PowerShell's PATH. The dashboard does not save form entries and
labels any estimate as exploratory rather than diagnostic.

The notebook detects older SciPy seed arguments, OneHotEncoder parameter names,
GroupKFold shuffle support, and SHAP plotting arguments. On older scikit-learn,
the all-row sensitivity expands the seeded primary training folds by response
pattern, preserving group separation. Sensitivity estimates may differ with this
fold layout; use the pinned versions for exact reproduction of the saved report.

`build_project.py` recovers the Phase 1 cells from the saved full notebook, executes all code cells
in a shared Python namespace, and writes their real stdout, warnings, errors
and PNG figures into standard nbformat-4 JSON. This avoids an nbformat/nbclient
dependency. On errors it saves a partial notebook with the traceback and stops.
Reruns overwrite generated results. Seeds are 42 where random state applies.

## Artifact map

| Path | Contents |
| --- | --- |
| `build_project.py` | Self-contained complete notebook source and executable build; it recovers the Phase 1 audit cells from the saved full notebook |
| `outputs/phase1/` | Audit report, column checks, actual label frequencies |
| `outputs/data/cleaned_survey_all_rows.csv` | All 2,613 records, 30 original columns, survey items encoded |
| `outputs/data/composite_indices_all_rows.csv` | All records with the five requested indices and optional 0–100 rescaling |
| `outputs/data/analysis_distinct_patterns.csv` | Primary distinct-pattern dataset |
| `outputs/data/*_local.csv` | Split positions, error cases and cluster assignments; local use only |
| `outputs/tables/` | Reliability, statistics, CV/test metrics, importance, cluster profiles and checks |
| `outputs/tables/measurement_parallel_analysis.csv` | Exploratory item-level parallel-analysis screen |
| `outputs/tables/training_only_hyperparameter_search.csv` | Training-CV tuning results; no test-set reuse |
| `outputs/tables/alternative_clustering_comparison.csv` | K-Means, Ward, and Gaussian-mixture internal comparisons |
| `outputs/figures/` | PNG and SVG plots |
| `outputs/models/best_regression_pipeline.joblib` | CV-selected pipeline fitted only on primary training rows |
| `outputs/models/best_model_a_pipeline.joblib` | Best training-CV usage/demographic predictor |
| `outputs/models/model_metadata.json` | Features, versions, mapping and limitations |
| `outputs/models/clustering_bundle.joblib` | Descriptive scaler, K-Means, PCA and centroid-based names |
| `outputs/results_summary.json` | Actual result summary and research answers |

## Method and leakage controls

The five dependence items construct the target and never enter predictors.
Model A uses five usage items plus demographics; Model B adds the three related
construct averages. A fixed 80/20 split and identical five-fold training CV
compare Dummy, Linear, Ridge, Random Forest, and Gradient Boosting regression.
Each estimator uses a training-fitted ColumnTransformer/Pipeline. Select by CV
RMSE before test evaluation; no tuning follows inspection of test errors or importance.

All exact copies of a full response pattern stay on one side of every split.
The primary analysis gives each pattern equal weight. The all-row model sensitivity
uses original multiplicities with pattern-grouped CV; these are different estimands.
No claim is made that patterns identify people. No department synonyms are merged.
The full analysis population is used for descriptive clustering only; clusters
are never predictors or classes in the supervised task.

## Privacy and sharing

The source survey license, recruitment documentation and permission to publish
responses were not supplied. `.gitignore` excludes raw CSVs, row-level exports,
trained artifacts, and the executed notebook (which contains row previews/error
positions). Aggregate tables/figures and code can be reviewed separately.
Before sharing a notebook, clear its outputs or remove respondent-level outputs,
and review dataset permissions. No data license is invented. Only load joblib
files you trust. Do not use this exploratory model to rank, diagnose or make
decisions about individual students.

Classification is intentionally omitted because there are no validated severity
cutoffs. `app.py` provides a local Streamlit dashboard; the HTML report remains
available as a static results gallery.
