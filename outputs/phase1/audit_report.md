# Phase 1 — Dataset audit

Project: AI Dependency and Cognitive Offloading Among University Students:
An Explainable Machine Learning Study

- File: `_(AI_Cognitive_Dependence_Survey_Data_BD_StudentsResponses) - Sheet1.csv`. The filename omits the `(1)` suffix in the original request.
- Shape: 2,613 rows × 30 columns; exact supplied schema and order match.
- Survey groups: five groups with five items each.
- Missing cells: 0.
- Blank text values: 0; surrounding-whitespace values: 0.
- Exact duplicate rows beyond first occurrences: 451 (17.26%).
- Rows in duplicate groups, including first occurrences: 493.
- Distinct complete response patterns: 2,162.
- Age: numeric `int64`, observed range 18–28.
- Age conversion failures: 0; basic Age-format violations: 0.
- All 25 survey columns contain exactly Strongly Disagree, Disagree, Neutral, Agree, and Strongly Agree.
- Unexpected nonmissing Likert responses: 0.
- Gender: Female, Male.
- University Type: Other, Private University, Public University.
- Academic Level: Postgraduate, Undergraduate.
- Department/Discipline: 35 distinct labels; see value_counts.csv for all labels and frequencies.

## Issues requiring attention in later phases

Matching records do not establish repeated respondents. No respondent identifier
or timestamp is available to resolve their origin. Keep all records in this audit.
Determine a documented policy before inference and splitting for machine learning;
overlapping repeated records could produce optimistic test performance.

Department names include possible abbreviation/full-name variants (CSE versus
Computer Science and Engineering; EEE versus Electrical and Electronic Engineering).
No categories have been merged without confirming their meaning.

No demographic codebook, eligibility rules, or sampling/provenance documentation
was supplied. Basic format checks cannot validate those properties or independence.
All responses are self-reports. No causal, cognitive-ability, or diagnostic
conclusions follow from this audit. Do not publish respondent-level records
or notebook previews without checking the dataset's sharing permissions.

Phase 1 is complete. No encoding, row deletion, index construction, or modeling
has been performed. The observed labels support the specified encoding in Phase 2.
