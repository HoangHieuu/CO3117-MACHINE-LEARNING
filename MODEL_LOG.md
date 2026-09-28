# Model log

Use one entry per model/major experiment. Keep the first attempt and post-reference revision identifiable in Git.

## Entry template

### Model and course topic

- Model:
- Syllabus chapter:
- Implementation depth (A — BUILD / B — DISSECT & MODIFY / C — APPLY & BENCHMARK):
- Status / checkpoint:

### Formulation and expected behavior

- Objective, probabilistic factorization, or decision rule:
- Assumptions and inductive bias:
- Expected failure modes:

### Data and implementation

- Data representation and preprocessing:
- First-attempt commit:
- Reference source and exact file/function/notebook:
- Code locations mapped to equations/algorithm steps:
- Modification/adaptation:
- Hyperparameters and selection procedure:

### Evidence

- Split and random seed:
- Macro-F1:
- Accuracy:
- Confusion matrix / diagnostic:
- Runtime and model/feature size:
- Focused experiment and result:
- Error or limitation analysis:
- Use-case suitability:
- Exam-ready explanation:

### Reproducibility

- Environment:
- Run command:
- Result/figure paths:
- Related weekly post and drill:

---

## Model entries

Add entries below as work begins. Do not fill results before running the corresponding experiment.

### W01–W04 catch-up — Decision Tree depth sensitivity

- Model: scikit-learn `DecisionTreeClassifier`, Gini criterion.
- Syllabus chapter: Chapter 2; release-time W04 catch-up.
- Implementation depth: B (reference dissection required); 
- Status / checkpoint: 2026-09-25; .
- Objective: greedy axis-aligned splits selected by weighted Gini impurity reduction; leaves predict the majority class.
- Assumptions and inductive bias: axis-aligned boundaries and greedy local split selection; deep trees can have high variance.
- Expected failure modes: underfitting at low depth; overfitting at large depth; subject leakage if rows are split without groups.
- Data representation and preprocessing: supplied 561-feature UCI HAR vectors; no learned preprocessing or scaling.
- Reference source: ML-From-Scratch `mlfromscratch/supervised_learning/decision_tree.py`, commit `a2806c6732eee8d27762edd6d864e0c179d8e9e8`; code mapping is in `docs/pre-release/PRE_RELEASE_CATCHUP.md`.
- Hyperparameters: `max_depth` in `{1, 2, 3, 5, 10, None}`; `min_samples_split=2`, `min_samples_leaf=1`, `random_state=42` held fixed.
- Split and seed: `GroupShuffleSplit(test_size=0.20, random_state=42)` within official training data; fit on 5,551 windows/16 subjects, validate on 1,801 windows/5 disjoint subjects. Official test files were not read.
- Focused experiment: vary only maximum depth. Best validation Macro-F1 among tested values is 0.885287 at depth 10; unrestricted tree reaches 1.0 training accuracy and 0.866588 validation Macro-F1.
- Full results and environment: `results/r0_tree_depth_validation.json`.
- Reproduction: `python experiments/part1_pre_midterm/run_tree_depth_experiment.py`.
- Limitation: one fixed validation population and a small depth grid; no official-test evaluation or claim of statistical superiority.

## R0 baseline — most-frequent activity

- Model: scikit-learn DummyClassifier(strategy="most_frequent").
- Purpose: simple lower-bound baseline to retain throughout the project.
- Data: fixed subject-aware split inside official UCI train; validation subjects 1, 3, 15, 25, 27. This baseline script reads no official test files. A separate structural audit checked test shapes, labels, subject IDs, and group separation; it did not score or tune a model.
- Fitted on: 5,551 windows from 16 subjects; predicts LAYING (class 6), the most frequent class in this fit subset.
- Evaluated on: 1,801 windows from 5 held-out validation subjects.
- Results: Macro-F1 0.051751; accuracy 0.183787. See results/r0_majority_baseline_validation.json for the confusion matrix, exact split, and environment versions.
- Limitation: this predicts one class for every window and is only a reference floor, not a useful activity recognizer.
- Reproduction: python experiments/part1_pre_midterm/run_majority_baseline.py
