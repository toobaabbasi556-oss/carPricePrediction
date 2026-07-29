"""Grouped split/CV integrity tests (SPEC §5).

A vehicle (make+model) must never straddle the train/test boundary or a CV fold, else
the estimator sees the same model at fit and validation time and leaks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pkcar.data.split import (
    cv_split_indices,
    grouped_train_test_split,
    make_group_key,
)


def _frame(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "make": rng.choice(["toyota", "honda", "suzuki", "kia"], n),
            "model": rng.choice(["corolla", "civic", "mehran", "alto", "sportage"], n),
        }
    )


def test_group_key_combines_make_and_model() -> None:
    df = pd.DataFrame({"make": ["Toyota"], "model": ["Corolla"]})
    assert make_group_key(df).iloc[0] == "toyota__corolla"


def test_train_test_have_no_shared_group() -> None:
    df = _frame()
    groups = make_group_key(df)
    train_idx, test_idx = grouped_train_test_split(df, test_size=0.2, seed=42)
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    assert train_groups.isdisjoint(test_groups)
    assert len(test_idx) > 0


def test_cv_folds_do_not_share_groups() -> None:
    df = _frame()
    groups = make_group_key(df)
    for train_idx, val_idx in cv_split_indices(df, n_splits=4):
        assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[val_idx]))
