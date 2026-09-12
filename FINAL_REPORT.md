# AI Dependency and Cognitive Offloading Among University Students: An Explainable Machine Learning Study

## Scope and central limitation
An exploratory analysis of self-reported survey responses. Weak internal consistency and large repeated-record effects limit construct interpretation. The indices do not objectively measure intelligence, memory, cognitive ability, or clinical conditions.

## Data and analysis population
The supplied CSV contains 2,613 rows and 30 columns, matching the specified schema, with no missing values. There are 451 exact duplicate rows beyond their first occurrences. The primary analysis gives each of 2,162 distinct complete response patterns equal weight. Full-row files are retained and sensitivity analyses preserve original multiplicity. Identical answers do not prove repeated respondents. Some full patterns occur 100 or 200 times. Possible department synonyms remain separate because no authoritative category mapping was supplied.

## Measurement and reliability
Five prespecified five-item averages are calculated after the exact Likert labels are mapped from 1 to 5. Raw Cronbach's alpha uses sample variances. These values are internal consistency evidence, not construct validity. The weak primary values mean these averages are descriptive exploratory summaries. No items were removed or recoded to improve alpha. [Tavakol and Dennick (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4205511/).

| construct | alpha_primary_distinct | alpha_all_rows |
| --- | --- | --- |
| AI_Usage_Index | 0.3331 | 0.4794 |
| AI_Dependency_Index | 0.3234 | 0.7062 |
| AI_Trust_Index | 0.2839 | 0.6100 |
| Cognitive_Offloading_Index | 0.5291 | 0.5563 |
| Academic_Decision_Index | 0.2207 | 0.5534 |

## Descriptive indices
Primary distinct-pattern summaries:

| construct | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AI_Usage_Index | 2162.0000 | 3.8198 | 0.6020 | 1.2000 | 3.4000 | 3.8000 | 4.2000 | 5.0000 |
| AI_Dependency_Index | 2162.0000 | 3.7224 | 0.6086 | 1.4000 | 3.4000 | 3.8000 | 4.0000 | 5.0000 |
| AI_Trust_Index | 2162.0000 | 3.7578 | 0.5825 | 1.8000 | 3.4000 | 3.8000 | 4.2000 | 5.0000 |
| Cognitive_Offloading_Index | 2162.0000 | 3.7544 | 0.7011 | 1.6000 | 3.4000 | 3.8000 | 4.2000 | 5.0000 |
| Academic_Decision_Index | 2162.0000 | 3.7790 | 0.5582 | 1.6000 | 3.4000 | 3.8000 | 4.2000 | 5.0000 |

## Statistical methods and six prespecified associations
Spearman rho is appropriate for monotonic relationships among tied Likert-derived averages. Two-sided asymptotic p-values assume independent records; Holm correction is applied across six tests per analysis population. Statistical evidence does not establish practical importance or causation. [SciPy Spearman reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).

| x | y | rho | p_value | p_holm |
| --- | --- | --- | --- | --- |
| AI_Usage_Index | AI_Dependency_Index | 0.139789 | 6.660e-11 | 1.332e-10 |
| AI_Dependency_Index | Cognitive_Offloading_Index | 0.284488 | 1.559e-41 | 7.797e-41 |
| AI_Trust_Index | AI_Dependency_Index | 0.268906 | 3.958e-37 | 1.583e-36 |
| AI_Dependency_Index | Academic_Decision_Index | 0.066834 | 0.001875 | 0.001875 |
| AI_Usage_Index | Cognitive_Offloading_Index | 0.182869 | 1.031e-17 | 3.094e-17 |
| AI_Trust_Index | Cognitive_Offloading_Index | 0.312740 | 2.894e-50 | 1.737e-49 |

## Repeated-record association sensitivity
All-row p-values may be optimistic if repetitions are repeated submissions. Different row weights change the estimand.

| sample | x | y | rho | p_holm |
| --- | --- | --- | --- | --- |
| primary_distinct | AI_Usage_Index | AI_Dependency_Index | 0.139789 | 1.332e-10 |
| primary_distinct | AI_Dependency_Index | Cognitive_Offloading_Index | 0.284488 | 7.797e-41 |
| primary_distinct | AI_Trust_Index | AI_Dependency_Index | 0.268906 | 1.583e-36 |
| primary_distinct | AI_Dependency_Index | Academic_Decision_Index | 0.066834 | 0.001875 |
| primary_distinct | AI_Usage_Index | Cognitive_Offloading_Index | 0.182869 | 3.094e-17 |
| primary_distinct | AI_Trust_Index | Cognitive_Offloading_Index | 0.312740 | 1.737e-49 |
| all_rows_sensitivity | AI_Usage_Index | AI_Dependency_Index | 0.389249 | 1.401e-94 |
| all_rows_sensitivity | AI_Dependency_Index | Cognitive_Offloading_Index | 0.358106 | 2.013e-79 |
| all_rows_sensitivity | AI_Trust_Index | AI_Dependency_Index | 0.545089 | 2.172e-201 |
| all_rows_sensitivity | AI_Dependency_Index | Academic_Decision_Index | 0.312840 | 4.000e-60 |
| all_rows_sensitivity | AI_Usage_Index | Cognitive_Offloading_Index | 0.232072 | 2.732e-33 |
| all_rows_sensitivity | AI_Trust_Index | Cognitive_Offloading_Index | 0.369583 | 9.057e-85 |

## Group comparisons
Group counts, ties, skew, quartiles and boxplots were inspected before tests. Rank tests suit the bounded discrete outcomes. Other university type has only four distinct patterns, so university omnibus H tests and small-group pairwise U tests use 9,999 seeded permutations. H uses its upper tail for any group difference; U comparisons are two-sided. No outcomes are missing, but respondent independence and null exchangeability remain unverified. [SciPy permutation reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html). Interpret results as distribution/rank differences, not necessarily median shifts. Rank-biserial effect is 2U/(n1*n2)-1 for the alphabetically first versus second group; rank epsilon squared is H/(N-1) for multiple groups. Holm adjustment covers six omnibus comparisons. Exploratory university pairwise results have their own six-test Holm correction in the exported table. [Mann–Whitney U reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html), [Kruskal–Wallis reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html).

| outcome | grouping | test | statistic | p_holm | effect_size | effect_name |
| --- | --- | --- | --- | --- | --- | --- |
| AI_Dependency_Index | Gender | Mann-Whitney U | 647816.500000 | 0.000054 | 0.109655 | rank_biserial_first_vs_second |
| AI_Dependency_Index | University Type | Kruskal-Wallis H (9999 permutations) | 15.938834 | 0.001000 | 0.007376 | rank_epsilon_squared |
| AI_Dependency_Index | Academic Level | Mann-Whitney U | 591853.500000 | 0.498154 | 0.016751 | rank_biserial_first_vs_second |
| Cognitive_Offloading_Index | Gender | Mann-Whitney U | 627901.000000 | 0.009015 | 0.075541 | rank_biserial_first_vs_second |
| Cognitive_Offloading_Index | University Type | Kruskal-Wallis H (9999 permutations) | 6.445771 | 0.068400 | 0.002983 | rank_epsilon_squared |
| Cognitive_Offloading_Index | Academic Level | Mann-Whitney U | 615825.500000 | 0.057980 | 0.057933 | rank_biserial_first_vs_second |

## Regression design and results
Model A uses the five usage items plus all demographics. Model B additionally uses the trust, offloading and academic-decision averages, making it an expanded associational model. All dependency items and score variants are excluded from predictors. The seed-42 80/20 split contains 1,729 training and 433 test patterns. Five identical shuffled training folds are used across models. Scalers, imputers and one-hot encoders are fitted inside pipelines on training data only. Dummy, linear, ridge, random forest and gradient boosting models provide constant, additive, regularized and nonlinear benchmarks. Parameters are fixed; no test-driven tuning is performed. CV RMSE selects the candidate before all held-out results are computed. The selected pipeline is B/Gradient Boosting; it is saved without refitting on the test set. [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

| feature_set | model | CV_RMSE_mean | CV_RMSE_sd | Test_MAE | Test_RMSE | Test_R2 | selected_by_cv |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | Random Forest | 0.5047 | 0.0273 | 0.4032 | 0.5341 | 0.2040 | False |
| A | Gradient Boosting | 0.5383 | 0.0238 | 0.4331 | 0.5428 | 0.1780 | False |
| A | Ridge | 0.5893 | 0.0241 | 0.4605 | 0.5815 | 0.0564 | False |
| A | Linear Regression | 0.5952 | 0.0248 | 0.4691 | 0.5887 | 0.0330 | False |
| A | Dummy | 0.6120 | 0.0186 | 0.4638 | 0.5987 | -0.0001 | False |
| B | Gradient Boosting | 0.4965 | 0.0301 | 0.3839 | 0.5036 | 0.2923 | True |
| B | Random Forest | 0.4989 | 0.0287 | 0.3852 | 0.5100 | 0.2744 | False |
| B | Ridge | 0.5392 | 0.0213 | 0.4230 | 0.5342 | 0.2038 | False |
| B | Linear Regression | 0.5442 | 0.0209 | 0.4264 | 0.5372 | 0.1949 | False |
| B | Dummy | 0.6120 | 0.0186 | 0.4638 | 0.5987 | -0.0001 | False |

## Regression multiplicity sensitivity
All copies of a response pattern stay in the same partition and GroupKFold fold. The primary candidates are frozen before this analysis. Because the split is by pattern, the multiplicity-weighted row counts do not necessarily remain 80/20.

| feature_set | model | n_train | n_test | CV_MAE_mean | CV_RMSE_mean | CV_R2_mean | Test_MAE | Test_RMSE | Test_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | Random Forest | 2172 | 441 | 0.5496 | 0.7141 | 0.1868 | 0.3990 | 0.5334 | 0.2491 |
| A | Dummy | 2172 | 441 | 0.6959 | 0.8555 | -0.2325 | 0.5223 | 0.6528 | -0.1248 |
| B | Gradient Boosting | 2172 | 441 | 0.5119 | 0.6610 | 0.3009 | 0.3788 | 0.5024 | 0.3338 |
| B | Dummy | 2172 | 441 | 0.6959 | 0.8555 | -0.2325 | 0.5223 | 0.6528 | -0.1248 |

## Error analysis
Residual is actual minus prediction. Positive values mean underprediction. There are 2 held-out predictions outside 1–5; predictions were not clipped. Largest-error cases are saved locally without demographics; they are not student diagnoses. Measurement noise, unobserved factors and model flexibility may contribute to errors, but these explanations are not established by the current analysis.

| quantity | count | mean | std | min | 5% | 25% | 50% | 75% | 95% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residual | 433.0000 | 0.0005 | 0.5042 | -1.8048 | -0.8268 | -0.2944 | 0.0260 | 0.2993 | 0.7960 | 1.3043 |
| absolute_error | 433.0000 | 0.3839 | 0.3263 | 0.0060 | 0.0226 | 0.1418 | 0.2993 | 0.5485 | 0.9856 | 1.8048 |

## Explainability
Permutation importance is computed on held-out rows using 20 seeded shuffles per original feature. Positive values indicate worsening RMSE after perturbation; standard deviations describe repeat variability, not confidence intervals. Correlated variables may share importance. Predictive importance does not establish a causal effect. [scikit-learn permutation documentation](https://scikit-learn.org/stable/modules/permutation_importance.html). SHAP 0.52.0 executed on 250 held-out patterns; additivity verified.

| feature | RMSE_increase_mean | RMSE_increase_sd |
| --- | --- | --- |
| Cognitive_Offloading_Index | 0.0589 | 0.0079 |
| AI_Trust_Index | 0.0420 | 0.0060 |
| Age | 0.0041 | 0.0021 |
| AI Usage [I use AI tools for assignment writing.] | 0.0022 | 0.0009 |
| Department/Discipline | 0.0020 | 0.0016 |
| University Type | 0.0019 | 0.0008 |
| Academic_Decision_Index | 0.0015 | 0.0009 |
| AI Usage [I use AI tools for study planning.] | 0.0008 | 0.0008 |
| Gender | 0.0006 | 0.0005 |
| AI Usage [I depend on AI tools to complete academic work faster.] | 0.0005 | 0.0006 |
| Academic Level | 0.0000 | 0.0000 |
| AI Usage [I use generative AI tools regularly for academic purposes.] | -9.274e-06 | 0.0004 |
| AI Usage [I use AI tools for research-related tasks.] | -3.972e-05 | 0.0001 |

## Clustering
K-Means uses only the five standardized indices, with k=2–6 and 20 initializations. The highest silhouette determines k; inertia is reported for context. PCA is visualization only, and all centroids are shown in original 1–5 units. Names were assigned only after printing centroids. The words lower/higher describe relative fitted profiles and do not define validated dependency categories. [scikit-learn silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis).

| k | inertia | silhouette |
| --- | --- | --- |
| 2 | 8631.0217 | 0.1693 |
| 3 | 7360.4436 | 0.2234 |
| 4 | 6480.8106 | 0.1874 |
| 5 | 5821.5851 | 0.1933 |
| 6 | 5266.3974 | 0.2016 |

| cluster | profile | n | AI_Usage_Index | AI_Dependency_Index | AI_Trust_Index | Cognitive_Offloading_Index | Academic_Decision_Index |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Intermediate dependency profile 2 of 3 | 1455 | 3.7442 | 3.8351 | 3.8656 | 3.8232 | 3.7348 |
| 1 | Relatively lower dependency profile | 500 | 3.6652 | 3.1244 | 3.1384 | 3.0836 | 3.6976 |
| 2 | Relatively higher dependency profile | 207 | 4.7246 | 4.3749 | 4.4966 | 4.8908 | 4.2860 |

## Supplementary enhancement analyses
Measurement diagnostics use item-level Spearman correlations, a 200-replicate parallel-analysis screen, and approximate-continuous FactorAnalysis. This does not replace ordinal psychometric validation. Suggested factor counts are:

| construct | suggested_factor_count |
| --- | --- |
| AI_Dependency_Index | 1 |
| AI_Trust_Index | 1 |
| AI_Usage_Index | 1 |
| Academic_Decision_Index | 2 |
| Cognitive_Offloading_Index | 1 |

Training-only RandomizedSearchCV tuned the Model A Random Forest and Model B Gradient Boosting candidates. The held-out test set was excluded, so these results are not compared or substituted for the primary test results.

| search | n_candidates | best_training_CV_RMSE | best_parameters |
| --- | --- | --- | --- |
| A_Random_Forest | 8 | 0.5029 | {"model__n_estimators": 180, "model__min_samples_leaf": 2, "model__max_features": 0.7, "model__max_depth": 6} |
| B_Gradient_Boosting | 8 | 0.4951 | {"model__subsample": 1.0, "model__n_estimators": 160, "model__min_samples_leaf": 12, "model__max_depth": 3, "model__learning_rate": 0.02} |

Alternative clustering uses the same standardized indices and reports internal criteria only. No method validates naturally occurring student types.

| method | k | silhouette | BIC |
| --- | --- | --- | --- |
| KMeans | 2 | 0.1693 | NA |
| KMeans | 3 | 0.2234 | NA |
| KMeans | 4 | 0.1874 | NA |
| KMeans | 5 | 0.1933 | NA |
| KMeans | 6 | 0.2016 | NA |
| Ward agglomerative | 2 | 0.3148 | NA |
| Ward agglomerative | 3 | 0.3201 | NA |
| Ward agglomerative | 4 | 0.1414 | NA |
| Ward agglomerative | 5 | 0.1346 | NA |
| Ward agglomerative | 6 | 0.1334 | NA |
| Gaussian mixture | 2 | 0.3166 | 20010.1899 |
| Gaussian mixture | 3 | 0.2525 | 14493.6509 |
| Gaussian mixture | 4 | 0.1942 | 17089.3462 |
| Gaussian mixture | 5 | 0.1663 | 13969.1908 |
| Gaussian mixture | 6 | 0.1752 | 11599.6494 |

## Research questions
### RQ1: Is greater academic AI usage associated with greater self-reported AI dependency?
Evidence of an association after Holm adjustment: Spearman rho=0.140 (positive), raw p=6.66e-11, Holm-adjusted p=1.332e-10, n=2,162. This is a cross-sectional association, not causation.

### RQ2: Is AI dependency associated with greater cognitive offloading?
Evidence of an association after Holm adjustment: Spearman rho=0.284 (positive), raw p=1.559e-41, Holm-adjusted p=7.797e-41, n=2,162. This is a cross-sectional association, not causation.

### RQ3: Is greater trust in AI associated with greater AI dependency?
Evidence of an association after Holm adjustment: Spearman rho=0.269 (positive), raw p=3.958e-37, Holm-adjusted p=1.583e-36, n=2,162. This is a cross-sectional association, not causation.

### RQ4: How is AI dependency associated with AI-influenced academic decision-making?
Evidence of an association after Holm adjustment: Spearman rho=0.067 (positive), raw p=0.001875, Holm-adjusted p=0.001875, n=2,162. This is a cross-sectional association, not causation.

### RQ5: Can AI usage patterns and demographic factors predict variation in AI Dependency Index?
The selected Model A improves on the dummy in both CV and held-out RMSE. A/Random Forest has CV RMSE=0.505; test MAE=0.403, RMSE=0.534, R²=0.204. The dummy test RMSE is 0.599; absolute RMSE improvement is 0.065 index units. Prediction refers to a weakly reliable item average in this sample; external usefulness is unestablished.

### RQ6: Which variables are most useful for predicting AI dependency?
In the CV-selected B/Gradient Boosting model, the largest positive held-out permutation effects are: Cognitive offloading (RMSE increase 0.0589 ± repeat SD 0.0079); AI trust (RMSE increase 0.0420 ± repeat SD 0.0060); Age (RMSE increase 0.0041 ± repeat SD 0.0021); use AI tools for assignment writing. (RMSE increase 0.0022 ± repeat SD 0.0009); Department/Discipline (RMSE increase 0.0020 ± repeat SD 0.0016). Rankings are model-specific, affected by correlated inputs, and do not identify causes.

### RQ7: Are there distinct student profiles based on the five item averages?
Among k=2–6, k=3 has the largest silhouette (0.223). PCA retains 60.8% of standardized variance in two dimensions. Mean subsample adjusted-Rand agreement is 0.910. The fitted profiles summarize these response patterns; the silhouette, centroid differences, weak scale reliability and lack of external validation do not establish naturally distinct student populations.

## Conclusions supported by this dataset
The four dependency-related research associations are positive, with primary Spearman coefficients from 0.067 to 0.284. The academic-decision association is particularly small despite its adjusted p-value. Usage items and demographics improve prediction over the training-mean baseline; the expanded Gradient Boosting model reaches test R²=0.292. Cognitive offloading and trust have the largest positive permutation effects in that fitted model. The 3-profile partition has silhouette 0.223, consistent with substantial overlap. Weak primary reliability and repeated-record sensitivity are central findings, so these results support an exploratory portfolio demonstration and further measurement/provenance investigation, not causal claims, validated student categories or individual assessments.

## Limitations and ethics
- Sampling frame, university identifiers, recruitment, response rate, consent and provenance are unavailable. The Bangladesh-student framing follows the supplied dataset description; geographic representativeness cannot be verified.
- Exact duplicates have unknown origin. Distinct-pattern analysis may remove independent coincident responses; full-row sensitivity may overweight repeated submissions. Neither policy verifies independence.
- Weak reliability undermines latent-construct interpretations and limits target measurement quality. Equal item weights and interval-like scoring are descriptive assumptions, not psychometric validation.
- Cross-sectional self-reports can share response styles and reporting biases. Temporal order, reverse association, confounding and causal effects cannot be resolved. No intelligence, memory or ability tests were administered.
- Model B uses conceptually overlapping self-report constructs. Prediction is not prospective early warning. A single held-out split and training CV do not establish external generalization or subgroup fairness.
- Pipelines prevent learned-preprocessing leakage and direct target leakage. Pattern separation prevents exact row overlap, but cannot identify repeated people whose answers differ. Exploratory EDA preceded modeling; feature sets and candidate settings follow the prespecified design and were not tuned to test results.
- K-Means always forms partitions; silhouette and descriptive stability do not validate distinct natural populations.
- No arbitrary supervised severity categories, diagnoses, impairment labels or causal recommendations are produced. Keep row-level exports, error cases, assignments and trained artifacts local until sharing rights are established.

## Future improvements
1. **Strengthen measurement.** All five primary item averages have weak internal consistency (alpha 0.221–0.529). Inspect item-level associations and conduct exploratory factor analysis in a future, appropriately designed measurement study to assess whether the items reflect one dimension or several. Revise and validate the questionnaire before treating these averages as established constructs.

2. **Resolve repeated-response provenance.** The dataset contains 451 exact duplicate rows beyond their first occurrences. Anonymous respondent identifiers or other privacy-preserving collection metadata would help distinguish repeat submissions from independent students giving identical answers.

3. **Tune models using training data only.** GridSearchCV or RandomizedSearchCV could tune Random Forest and Gradient Boosting hyperparameters inside cross-validation. A final test set must remain untouched until model selection is complete.

4. **Study prediction error, especially at score extremes.** The selected Gradient Boosting model has held-out RMSE=0.504 and R²=0.292; its predictions are concentrated toward the target's central range. Future work could test pre-registered feature engineering, additional behavioral variables, alternative regressors, and larger independently collected samples.

5. **Validate profile structure.** k=3 has the highest tested silhouette (0.223), but the value indicates overlap. Compare K-Means with hierarchical clustering and Gaussian mixture models, then test whether comparable profiles replicate in an independent sample.

6. **Collect longitudinal and behavioral evidence.** Multi-semester data and appropriate behavioral measures, such as logged AI-use frequency or independent problem-solving tasks, could clarify temporal patterns. They would still require careful consent, privacy controls, and causal design.

7. **Externally validate the study.** Test the measurement structure and frozen model in independently collected samples from different universities, disciplines, regions, and countries. The present sample's representativeness cannot be verified.

8. **Consider an interactive portfolio demonstration only after validation.** A FastAPI backend and React frontend could present an exploratory model estimate and model explanation. It should minimise data collection, protect respondent privacy, and clearly state that the output is neither a diagnosis nor a validated assessment.

## Reproducibility and deliverables
Run `python build_project.py` from the repository root. The script executes every notebook code cell in order, captures real outputs and figures, and stops on errors. It uses a standard-library notebook writer because nbformat/nbclient are unavailable locally. Core versions are pinned in requirements.txt; SHAP is optional and its execution status is reported above. The notebook includes the source code for every analysis. See README.md for the artifact map and data policy.
