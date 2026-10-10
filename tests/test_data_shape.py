"""Проверки формы данных и корректности split."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config, load_config
from src.data import load_clean, split_indices


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="module")
def data(cfg: Config) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = load_clean(cfg)
    train_idx, test_idx = split_indices(df, cfg)
    return df, train_idx, test_idx


def test_split_sizes(data, cfg: Config) -> None:
    df, train_idx, test_idx = data
    assert len(train_idx) + len(test_idx) == len(df)
    assert len(test_idx) == pytest.approx(len(df) * cfg.test_size, abs=1)


def test_train_test_indices_disjoint(data) -> None:
    _, train_idx, test_idx = data
    assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))


def test_churn_rate_similar(data, cfg: Config) -> None:
    df, train_idx, test_idx = data
    train_rate = float(df.iloc[train_idx][cfg.target].mean())
    test_rate = float(df.iloc[test_idx][cfg.target].mean())
    assert abs(train_rate - test_rate) < 0.02


def test_target_is_binary(data, cfg: Config) -> None:
    df, _, _ = data
    values = set(df[cfg.target].unique().tolist())
    assert values <= {0, 1}


def test_id_column_dropped_and_no_missing_target(data, cfg: Config) -> None:
    df, _, _ = data
    assert cfg.id_column not in df.columns
    assert df["TotalCharges"].isna().sum() == 0
