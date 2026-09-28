#!/usr/bin/env python3
"""Compare DecisionTree max_depth values on the frozen UCI HAR validation split.

This experiment loads only the official training features, labels, and subjects.
It never reads the official test files.
"""

from __future__ import annotations

import json
import platform
from datetime import date
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "human+activity+recognition+using+smartphones" / "UCI HAR Dataset"
OUTPUT = ROOT / "results" / "r0_tree_depth_validation.json"
CLASS_IDS = np.arange(1, 7)
DEPTHS: tuple[int | None, ...] = (1, 2, 3, 5, 10, None)


def main() -> None:
    train_dir = DATA_ROOT / "train"
    X = np.loadtxt(train_dir / "X_train.txt")
    y = np.loadtxt(train_dir / "y_train.txt", dtype=int)
    subjects = np.loadtxt(train_dir / "subject_train.txt", dtype=int)

    if X.ndim != 2 or y.ndim != 1 or subjects.ndim != 1:
        raise ValueError("Expected 2-D X and aligned 1-D labels/subject IDs.")
    if not (len(X) == len(y) == len(subjects)):
        raise ValueError("Training features, labels, and subjects are not aligned.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    fit_idx, validation_idx = next(splitter.split(X, y, groups=subjects))
    fit_subjects = sorted(int(v) for v in np.unique(subjects[fit_idx]))
    validation_subjects = sorted(int(v) for v in np.unique(subjects[validation_idx]))
    if set(fit_subjects) & set(validation_subjects):
        raise RuntimeError("Subject leakage detected between fit and validation.")

    rows = []
    for depth in DEPTHS:
        model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=depth,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        )
        model.fit(X[fit_idx], y[fit_idx])
        fit_pred = model.predict(X[fit_idx])
        validation_pred = model.predict(X[validation_idx])
        rows.append(
            {
                "max_depth": depth,
                "tree_depth": int(model.get_depth()),
                "leaf_count": int(model.get_n_leaves()),
                "train_macro_f1": float(
                    f1_score(y[fit_idx], fit_pred, labels=CLASS_IDS, average="macro", zero_division=0)
                ),
                "train_accuracy": float(accuracy_score(y[fit_idx], fit_pred)),
                "validation_macro_f1": float(
                    f1_score(y[validation_idx], validation_pred, labels=CLASS_IDS, average="macro", zero_division=0)
                ),
                "validation_accuracy": float(accuracy_score(y[validation_idx], validation_pred)),
            }
        )

    result = {
        "date": date.today().isoformat(),
        "dataset": "UCI HAR Dataset Version 1.0 (UCI record 240)",
        "experiment": "DecisionTreeClassifier max_depth sensitivity",
        "criterion": "gini",
        "other_parameters_held_fixed": {
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        },
        "validation_policy": "GroupShuffleSplit(test_size=0.20, random_state=42) on official training subjects",
        "fit_subjects": fit_subjects,
        "validation_subjects": validation_subjects,
        "fit_windows": int(len(fit_idx)),
        "validation_windows": int(len(validation_idx)),
        "rows": rows,
        "official_test_files_read": False,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved result: {OUTPUT}")


if __name__ == "__main__":
    main()
