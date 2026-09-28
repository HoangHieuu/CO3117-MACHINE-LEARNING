# Data provenance and frozen protocol

## Source and use case

- Dataset: Human Activity Recognition Using Smartphones, UCI HAR Dataset Version 1.0.
- Canonical source: UCI Machine Learning Repository, dataset record 240.
- Dataset page: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- DOI: https://doi.org/10.24432/C54S4K
- Retrieved locally: 2026-09-25.
- Local extracted root: human+activity+recognition+using+smartphones/UCI HAR Dataset/
- Integrity manifest: UCI_HAR_MANIFEST.sha256.
- Use case: predict a person's current physical activity from smartphone inertial measurements.
- Target: activity ID mapped by activity_labels.txt: 1 WALKING, 2 WALKING_UPSTAIRS, 3 WALKING_DOWNSTAIRS, 4 SITTING, 5 STANDING, 6 LAYING.
- Unit of prediction: one 2.56-second window (128 sensor readings, 50% overlap).

The archive README identifies this as Version 1.0. It contains 30 subjects, six activities, 561 precomputed time/frequency features per window, aligned subject IDs, and inertial signal files. The UCI split is subject-disjoint: 21 subjects in the official training set and 9 subjects in the official test set. The extracted source files were checked for the expected row and feature counts; see the R0 baseline record for validation details.

Keep the downloaded/extracted source data local. The repository ignores the downloaded directory; commit provenance, scripts, and result summaries only.

## Frozen split, metrics, and seeds

- Official final test: use the supplied UCI test split (2,947 windows from 9 subjects) only for the final comparison. It has not been read by the R0 baseline.
- Model-development pool: use the supplied UCI training split (7,352 windows from 21 subjects).
- Fixed validation policy: GroupShuffleSplit with test_size=0.20 and random_state=42, grouping by subject ID.
- Validation subjects: 1, 3, 15, 25, 27 (1,801 windows).
- Model-fitting subjects: 5, 6, 7, 8, 11, 14, 16, 17, 19, 21, 22, 23, 26, 28, 29, 30 (5,551 windows).
- Primary metric: Macro-F1.
- Secondary metrics: accuracy and confusion matrix, with class order 1 through 6.
- Randomness: use seed 42 for data splitting and as the default seed for stochastic model procedures; record any model-specific seed changes.
- Baseline: most-frequent-class DummyClassifier fitted on the model-fitting subset and evaluated on the fixed validation subjects.

Do not move validation subjects, alter the official UCI train/test assignment, or tune on the official test set. Fit any learned preprocessing, feature selection, PCA/LDA, or other representation transform on the model-fitting subset only; apply the fitted transform to validation and, once unsealed for final evaluation, test.

Integrity-audit note: before freezing this protocol, the downloaded test files were read only to check row/column shapes, labels, subject IDs, and the train/test subject separation. No model was fitted, tuned, or scored on the official test split. The baseline script reads only the official training files. Keep all future model selection on the fixed training/validation split.

## Representations

### Static model input

Use X_train/X_test as the supplied 561-feature vectors, with y_train/y_test as activity IDs and subject_train/subject_test as group IDs. The UCI README says the supplied feature vectors are already bounded within [-1, 1]. Any additional scaler or learned transform must still be fitted only on the model-fitting subset.

### Sequential/structured view for HMM or CRF

The source supplies fixed-width, overlapping windows and nine aligned Inertial Signals files for each split. Preserve source row order and align every window with its activity and subject ID. Build sequences only within one subject and one original split; never create a transition across subjects or across train/validation/test boundaries. Do not shuffle rows before sequence construction.

The archive does not provide an explicit timestamp or session identifier for every window. Treat original row order within each split as a sequence-order proxy, document this assumption in the HMM/CRF experiment, and discuss the resulting limitation. Check how subject and activity boundaries appear before defining sequence segments. Do not claim that the inferred sequence is continuous free-living behavior.

## Local file validation

Expected source files include:

- README.txt, features_info.txt, features.txt, and activity_labels.txt.
- X, y, and subject files for both train and test.
- Nine Inertial Signals files for each split: body_acc, body_gyro, and total_acc across x/y/z.

Validated locally: X_train is 7,352 × 561; X_test is 2,947 × 561; the label/subject rows align; each inertial file has the matching number of windows and 128 readings per window; all six classes occur in both supplied splits; train/test subjects do not overlap.

## Reproduction

- Download the UCI HAR Dataset ZIP from the UCI dataset page above and extract it under data/human+activity+recognition+using+smartphones/.
- Baseline command from the repository root: python experiments/part1_pre_midterm/run_majority_baseline.py
- The command reads only the UCI training labels and subject IDs, writes results/r0_majority_baseline_validation.json, and appends its validation result to results/metrics.csv. It does not load the official test data.
- Environment: Python, NumPy, and scikit-learn versions are recorded in the result JSON.
- File SHA-256 values for the extracted source files are recorded in UCI_HAR_MANIFEST.sha256.
