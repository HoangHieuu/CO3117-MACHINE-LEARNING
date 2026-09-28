#!/usr/bin/env python3
"""Run a majority-class baseline on a fixed subject-aware validation split.

The official UCI test files are deliberately never read by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "human+activity+recognition+using+smartphones"
    / "UCI HAR Dataset"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results" / "r0_majority_baseline_validation.json"
METRICS_CSV = REPOSITORY_ROOT / "results" / "metrics.csv"
CLASS_IDS = np.arange(1, 7)
CLASS_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    y_path = data_root / "train" / "y_train.txt"
    subject_path = data_root / "train" / "subject_train.txt"
    if not y_path.is_file() or not subject_path.is_file():
        raise FileNotFoundError(
            f"Expected UCI training label and subject files under {data_root}"
        )

    y = np.loadtxt(y_path, dtype=int)
    subjects = np.loadtxt(subject_path, dtype=int)
    if y.ndim != 1 or subjects.ndim != 1 or len(y) != len(subjects):
        raise ValueError("Training labels and subject IDs must be aligned 1-D arrays.")
    if not set(np.unique(y)).issubset(set(CLASS_IDS)):
        raise ValueError("Unexpected activity IDs; expected IDs 1 through 6.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.validation_fraction,
        random_state=args.seed,
    )
    fit_indices, validation_indices = next(
        splitter.split(np.zeros((len(y), 1)), y, groups=subjects)
    )
    fit_subjects = sorted(int(value) for value in np.unique(subjects[fit_indices]))
    validation_subjects = sorted(
        int(value) for value in np.unique(subjects[validation_indices])
    )
    if set(fit_subjects) & set(validation_subjects):
        raise RuntimeError("Subject leakage detected between fit and validation.")

    model = DummyClassifier(strategy="most_frequent")
    fit_features = np.zeros((len(fit_indices), 1), dtype=float)
    validation_features = np.zeros((len(validation_indices), 1), dtype=float)

    fit_start = perf_counter()
    model.fit(fit_features, y[fit_indices])
    fit_seconds = perf_counter() - fit_start

    inference_start = perf_counter()
    predictions = model.predict(validation_features)
    inference_seconds = perf_counter() - inference_start

    matrix = confusion_matrix(y[validation_indices], predictions, labels=CLASS_IDS)
    result = {
        "dataset": "UCI HAR Dataset Version 1.0 (UCI record 240)",
        "model": "DummyClassifier(strategy=most_frequent)",
        "purpose": "majority-class lower-bound baseline",
        "validation_policy": "GroupShuffleSplit on official training subjects",
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "fit_subjects": fit_subjects,
        "validation_subjects": validation_subjects,
        "fit_windows": int(len(fit_indices)),
        "validation_windows": int(len(validation_indices)),
        "predicted_activity_id": int(model.classes_[np.argmax(model.class_prior_)]),
        "predicted_activity_name": CLASS_NAMES[
            int(model.classes_[np.argmax(model.class_prior_)])
        ],
        "macro_f1": float(
            f1_score(
                y[validation_indices],
                predictions,
                labels=CLASS_IDS,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y[validation_indices], predictions)),
        "confusion_matrix_labels": [int(value) for value in CLASS_IDS],
        "confusion_matrix_activity_names": [
            CLASS_NAMES[int(value)] for value in CLASS_IDS
        ],
        "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "baseline_reads_official_test_files": False,
        "test_structural_audit_before_baseline": True,
        "test_audit_scope": (
            "row/column shapes, label IDs, subject IDs, and train/test group "
            "disjointness; no test model scoring or tuning"
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_metrics_row(result)

    print(json.dumps(result, indent=2))
    print(f"Saved result: {args.output}")
    print(f"Updated metric row: {METRICS_CSV}")


def write_metrics_row(result: dict[str, object]) -> None:
    header = [
        "part",
        "course_week",
        "model",
        "split",
        "seed",
        "macro_f1",
        "accuracy",
        "train_seconds",
        "inference_seconds",
        "notes",
    ]
    row = [
        "R0",
        "R0",
        "DummyClassifier-most_frequent",
        "subject_group_validation",
        str(result["seed"]),
        f"{result['macro_f1']:.12f}",
        f"{result['accuracy']:.12f}",
        f"{result['fit_seconds']:.6f}",
        f"{result['inference_seconds']:.6f}",
        "Validation only; baseline reads train files; test structural audit only; see result JSON",
    ]
    existing_rows: list[list[str]] = []
    if METRICS_CSV.exists():
        with METRICS_CSV.open(newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            existing_header = next(reader, None)
            if existing_header and existing_header != header:
                raise ValueError(f"Unexpected metrics.csv header: {existing_header}")
            existing_rows = [
                old_row
                for old_row in reader
                if not (
                    len(old_row) > 3
                    and old_row[2] == row[2]
                    and old_row[3] == row[3]
                )
            ]

    with METRICS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(existing_rows)
        writer.writerow(row)


if __name__ == "__main__":
    main()
