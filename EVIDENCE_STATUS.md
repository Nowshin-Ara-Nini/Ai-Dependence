# Evidence Status and Evaluation Gaps

This note separates results supported by released artifacts from analyses that have **not** been run. A missing diagnostic is not a negative result, and an internal-split result is not external validation.

## Supported by released artifacts

| Topic | Evidence | Bounded interpretation |
| --- | --- | --- |
| Data audit | 2,613 complete rows; 451 exact duplicates beyond the first; 2,162 distinct patterns in primary analysis | Exact duplicates exist, but their respondent-level meaning is unknown because IDs are unavailable. |
| Measurement | Primary alphas: Dependency 0.323; related indices 0.221–0.529 | Internal consistency is weak to moderate at best. The composites are exploratory rather than validated scales. |
| Internal prediction | Model B/Gradient Boosting: held-out MAE 0.384, RMSE 0.504, R² 0.292 (433 test rows) | It improves on a mean-prediction baseline in this internal split, but substantial target variation is unexplained. |
| Duplicate sensitivity | All-row grouped sensitivity: MAE 0.379, RMSE 0.502, R² 0.334 (441 test rows) | The reported RMSE is similar under this alternate duplicate treatment; it does not identify what duplicate rows represent. |
| Feature importance | Held-out permutation importance ranks Cognitive Offloading and AI Trust highest | It describes model reliance; it is not causal evidence. |

## Duplicate-sensitivity comparison

| Evaluation | Rows analysed | Test rows | Test MAE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary: distinct complete response patterns | 2,162 | 433 | 0.384 | 0.504 | 0.292 |
| Sensitivity: all rows, identical predictor/target patterns grouped before split | 2,613 | 441 | 0.379 | 0.502 | 0.334 |

The rounded RMSE difference is 0.001 index units. The evaluations use different sample sizes and test sets, so this is not a paired statistical test or proof of robustness. Report it as a limited sensitivity analysis.

## Not yet assessed

### Calibration

This continuous-score model should be assessed with calibration-in-the-large (mean prediction minus mean observed target), a calibration intercept/slope, and an observed-versus-predicted plot in pre-specified quantile bins. None of these results is released. Existing residual and actual-versus-predicted figures do not by themselves establish calibration.

### Subgroup performance

The release contains descriptive and group-comparison analyses but no pre-specified held-out prediction metrics by Gender, Age band, University Type, Academic Level, or Department/Discipline. It therefore makes no claim that errors are comparable across groups. Small categories should be reported as insufficient for evaluation rather than treated as evidence of equal performance.

### Generalisation and measurement validity

There is no external, temporal, institutional, or prospective validation. The survey is self-reported and cross-sectional. Weak internal consistency means better predictive metrics alone would not establish a valid measure of AI dependency.

## Minimum next evaluation release

When row-level data can be used with appropriate permission, publish a frozen evaluation artifact containing:

1. the deduplication rule and counts before/after it;
2. a split identifier preventing overlapping response patterns across train and test;
3. overall MAE, RMSE, R², calibration intercept/slope, and bootstrap 95% confidence intervals;
4. a calibration table/plot using pre-specified prediction quantiles;
5. the same metrics for pre-specified, adequately sized subgroups, with counts and uncertainty intervals; and
6. an external or later-collected validation result kept separate from model selection.

Until then, the Streamlit app should remain an exploratory educational demo and should not display threshold labels such as “low”, “medium”, or “high risk”.
