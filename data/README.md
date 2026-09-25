# Data provenance and protocol

## Proposed dataset

- Source: UCI Human Activity Recognition Using Smartphones.
- URL: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Exact version/retrieval date: TODO — record before R0 is frozen.
- Prediction target: TODO — confirm activity label and class mapping.
- Use case: predict a person's current physical activity from smartphone inertial measurements.

## Split and evaluation

- Split policy: TODO — use subject/group-aware splitting when repeated measurements share a person.
- Primary metric: Macro-F1.
- Secondary metrics: accuracy and confusion matrix.
- Random seeds: TODO.
- Test population: TODO — keep sealed for final comparison.

## Preprocessing and representations

Document every scaler, encoder, feature-selection step, dimensionality-reduction transform, and sequential/structured view. Fit transforms on training data only, then apply them to validation/test data.

## Reproduction

- Data acquisition steps: TODO.
- Preparation command/script: TODO.
- Expected file layout: TODO.
