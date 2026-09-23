# Model Card: AI Dependency Index Estimate (Model B)

## Summary

This is a research demonstration that estimates an exploratory AI Dependency Index from self-reported questionnaire responses. It is not a validated psychometric scale, clinical assessment, diagnostic tool, or measure of intelligence, memory, or ability.

The saved model is a Gradient Boosting regression pipeline selected using five-fold cross-validation on the training split. Its output should be described as an *exploratory estimated item-average score*, not as a finding that an individual is dependent on AI.

## Intended use and prohibited use

Appropriate uses are portfolio demonstration, teaching, and exploratory research in a context similar to this sample. It must not be used to rank, screen, diagnose, discipline, admit, employ, or otherwise make a high-impact decision about a person.

## Model and feature contract

| Field | Released value |
| --- | --- |
| Target | `AI_Dependency_Index` |
| Target definition | Mean of five cognitive-dependence Likert items (1–5) |
| Feature set | Model B |
| Estimator | Gradient Boosting within the saved preprocessing pipeline |
| Training/test rows | 1,729 / 433 |
| Selection | Minimum mean five-fold training-CV RMSE; MAE tie-breaker |
| Random state | 42 |

The pipeline accepts 13 inputs: five AI-usage items; Gender, Age, University Type, Academic Level, and Department/Discipline; and the AI Trust, Cognitive Offloading, and Academic Decision-Making indices. The five items used to construct the target are excluded from predictors, preventing direct target leakage.

`outputs/models/model_metadata.json` is the source of truth for the exact feature names, Likert mapping, dependency versions, and feature contract.

## Data and measurement context

The project analysed a cross-sectional self-report survey of university students in Bangladesh. It contains 2,613 complete rows and 30 columns. The primary analysis retained 2,162 distinct complete response patterns after removing 451 exact duplicate rows beyond the first copy. Respondent IDs were unavailable, so duplicates cannot be classified as repeat submissions or different respondents with identical answers.

The target is an exploratory composite. In the primary distinct-pattern analysis its Cronbach's alpha was 0.323. Related composite alphas were AI Usage 0.333, AI Trust 0.284, Cognitive Offloading 0.529, and Academic Decision-Making 0.221. These values limit measurement confidence; they do not validate the constructs or support individual-level interpretation.

## Performance evidence

Evaluation occurred once on the 433-row held-out split, after model selection on training-fold cross-validation.

| Metric | Selected Model B |
| --- | ---: |
| Test MAE | 0.384 |
| Test RMSE | 0.504 |
| Test R² | 0.292 |

The test R² indicates that substantial target variation remains unexplained. These figures measure performance on one internal split of this study sample only; they do not show performance for other cohorts, institutions, times, or response modes.

## Duplicate-sensitivity evidence

An existing all-rows sensitivity analysis includes all 2,613 records, groups identical predictor/target patterns before splitting, and evaluates 2,172 training and 441 test rows. For Model B it reports test MAE 0.379, RMSE 0.502, and R² 0.334. The RMSE is close to the primary result (0.504); R² is modestly higher (0.334 versus 0.292).

This is **not** a duplicate-provenance test, independent replication, or guarantee that duplicates cannot affect other estimates. The source table is `outputs/tables/model_all_rows_grouped_sensitivity.csv`; see [EVIDENCE_STATUS.md](EVIDENCE_STATUS.md) for the bounded interpretation.

## Interpretability

Held-out permutation importance shows which inputs this fitted model used to reduce error in this test set. Cognitive Offloading and AI Trust had the largest reported effects. With correlated self-report constructs, importance is model-specific and does not identify causal effects.

## What has not been established

- External, temporal, geographic, or institutional validation.
- Calibration-in-the-large, calibration slope, or calibration curves on a held-out or external sample.
- Prediction-error or calibration comparisons for demographic or academic subgroups.
- Measurement invariance, construct validity, test-retest reliability, or a meaningful decision threshold.
- Safe use in consequential decisions.

## Before any research extension

Pre-register the target, feature set, split policy, and subgroup definitions; document a respondent-level duplicate rule; evaluate an untouched external or later-collected sample; report MAE, RMSE, R², calibration metrics, and confidence intervals; then reassess the survey measurement model. Any change to data, target construction, preprocessing, features, or estimator requires a new card and evaluation.
