# CO3117 Machine Learning — Individual Longitudinal Assignment

Repository for the individual CO3117 assignment. The project follows one prediction problem across multiple model families so their assumptions, learning procedures, behavior, and limitations can be compared under a common experimental protocol.

## Project status

- Assignment checkpoint: R0 setup in progress.
- Dataset/use case: proposed UCI Human Activity Recognition Using Smartphones; confirm and freeze the exact version and protocol before experiments.
- Primary metric: Macro-F1 for multiclass classification, with accuracy and a confusion matrix as secondary results.
- Submission parts: Part I (14 October 2026) and Part II (two calendar days before the official final examination date).

Update this section when the dataset, split, metric, and random seeds are frozen. Do not commit raw or processed dataset files unless the course policy explicitly requires it.

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

See the assignment specification and course LMS for authoritative requirements and cutoff times.
