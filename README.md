# CO3117 Machine Learning — Individual Longitudinal Assignment

Repository for the individual CO3117 assignment. The project follows one prediction problem across multiple model families so their assumptions, learning procedures, behavior, and limitations can be compared under a common experimental protocol.

## Project status

- Assignment checkpoint: R0 data protocol and validation baseline prepared in the working tree; the handwritten catch-up/baseline diagnostic and release-baseline tag remain pending.
- Dataset/use case: UCI Human Activity Recognition Using Smartphones, dataset version 1.0 (UCI record 240; DOI 10.24432/C54S4K). Predict one of six activities from smartphone inertial measurements.
- Primary metric: Macro-F1 for multiclass classification, with accuracy and a confusion matrix as secondary results.
- Evaluation split: preserve UCI's subject-disjoint official test set. Create one fixed, subject-aware validation split from official training subjects using GroupShuffleSplit with test_size=0.20 and random_state=42.
- Submission parts: Part I (14 October 2026) and Part II (two calendar days before the official final examination date).

The validation subjects are 1, 3, 15, 25, and 27; the remaining 16 official training subjects form the model-fitting subset. No official-test model score or tuning was performed. A structural integrity audit inspected test-file shapes, subject IDs, and class IDs; see data/README.md. Do not commit raw or processed dataset files.

## Repository map

- PROGRESS.md — weekly dashboard and checkpoint links.
- MODEL_LOG.md — per-model assumptions, implementation, experiments, and results.
- AI_USE.md and REFERENCES.md — AI-use disclosure and sources.
- SUBMISSION_PART1.md and SUBMISSION_PART2.md — submission checklists.
- data/ — dataset provenance and protocol documentation.
- docs/ — catch-up and weekly theory-to-code posts.
- exercises/ and exam/ — first attempts, corrections, and exam preparation.
- src/ — shared data/metric utilities and model implementations.
- experiments/ and results/ — experiment records, metrics, and figures.
- report/ — Part I summary and final report.

## Local setup

Requires Python 3.10 or later.

1. Create and activate a virtual environment.
2. Install the project and development tools with: pip install -e ".[dev]"
3. Run tests with: python -m pytest

Record the actual Python/package versions and random seeds used for submitted experiments. The setup commands are instructions only; no environment has been installed as part of this scaffold.

## Working rules

- Keep the dataset source/version, prediction target, split policy, test population, primary metric, and random seeds fixed after R0.
- Fit preprocessing and dimensionality-reduction steps on training data only.
- Keep the final test population sealed during tuning.
- Preserve the first attempt before inspecting reference implementations or using AI for that task; commit the later corrected state separately.
- Cite external code precisely and explain the connection between equations and code.
- Do not backdate work or rewrite a checkpoint after it has been reviewed.
