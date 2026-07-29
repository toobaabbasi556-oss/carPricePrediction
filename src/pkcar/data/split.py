"""Grouped-aware train/test split and CV folds (SPEC §5).

**Why grouped, not KFold, and not temporal.** SPEC §5 warns: "the same model recurs
across rows and will leak." Popular vehicles appear hundreds of times (civic 766,
corolla 737); near-duplicate listings of the *same* model differ only in noise. A naive
random split scatters a model's listings across train and test, letting the estimator
memorize a model's price level and inflating scores. So we group by **vehicle
(make + model)**: every listing of a given model lands entirely in train or entirely in
test / a single fold.

A true *temporal* split is not possible here: the data is a single snapshot with no
listing-posted date (SPEC §11), and ``scrape_date`` only encodes when our resumable
crawler happened to run. Grouped CV is the honest, leakage-safe choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

GROUP_SEP = "__"


def make_group_key(df: pd.DataFrame) -> pd.Series:
    """Vehicle group key ``make__model`` — the unit kept intact across splits (SPEC §5)."""
    make = df["make"].astype(str).str.strip().str.lower()
    model = df["model"].astype(str).str.strip().str.lower()
    return make.str.cat(model, sep=GROUP_SEP)


def grouped_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    groups: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Held-out test split with no vehicle group shared across train/test (SPEC §5, §7).

    Returns positional ``(train_idx, test_idx)`` arrays. The held-out test set is
    touched only for final reporting, never during CV/tuning (SPEC §7).
    """
    grp = groups if groups is not None else make_group_key(df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=grp))
    return train_idx, test_idx


def grouped_cv(n_splits: int) -> GroupKFold:
    """Grouped CV splitter (SPEC §5). Pass ``groups=make_group_key(X)`` to ``.split``.

    ``GroupKFold`` guarantees a vehicle's listings never straddle a fold boundary, so
    every transform fit inside a fold (SPEC §5) sees only unseen models at validation.
    """
    return GroupKFold(n_splits=n_splits)


def cv_split_indices(
    df: pd.DataFrame,
    n_splits: int,
    groups: pd.Series | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize grouped CV fold indices for inspection/tests (SPEC §5)."""
    grp = groups if groups is not None else make_group_key(df)
    return list(grouped_cv(n_splits).split(df, groups=grp))
