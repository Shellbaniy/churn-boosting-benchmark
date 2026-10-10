"""Детерминизм: при одном seed два обучения дают идентичные predict_proba."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipelines import MODEL_NAMES, available_variants, build_pipeline

# Малое число деревьев только для скорости теста; на дефолты этапа 4 это не влияет.
SMALL_PARAMS = {
    "logreg": None,
    "xgb": {"n_estimators": 10},
    "lgbm": {"n_estimators": 10},
    "catboost": {"iterations": 10},
}

CASES = [(model, variant) for model in MODEL_NAMES for variant in available_variants(model)]


@pytest.mark.parametrize("model_name,variant", CASES)
def test_same_seed_identical_predict_proba(model_name, variant, small_train, cfg) -> None:
    X, y = small_train
    params = SMALL_PARAMS[model_name]

    first = build_pipeline(model_name, variant, params, cfg).fit(X, y).predict_proba(X)[:, 1]
    second = build_pipeline(model_name, variant, params, cfg).fit(X, y).predict_proba(X)[:, 1]

    delta = float(np.abs(first - second).max())
    print(f"{model_name}/{variant}: строк = {len(first)}, max |Δp| = {delta:.3e}")
    assert np.array_equal(first, second)


@pytest.mark.parametrize("model_name,variant", CASES)
def test_same_seed_identical_learned_preprocessing(model_name, variant, small_train, cfg) -> None:
    X, y = small_train
    params = SMALL_PARAMS[model_name]
    step = "preprocess" if variant == "A" else "categorical"

    first = build_pipeline(model_name, variant, params, cfg).fit(X, y).named_steps[step].transform(X)
    second = build_pipeline(model_name, variant, params, cfg).fit(X, y).named_steps[step].transform(X)

    print(f"{model_name}/{variant}: форма препроцессинга = {first.shape}, тип = {type(first).__name__}")
    if isinstance(first, pd.DataFrame):
        pd.testing.assert_frame_equal(first, second)
    else:
        assert np.array_equal(first, second)


def test_determinism_check_can_detect_differences(small_train, cfg) -> None:
    """Контроль: тест сравнения не пустой — другое число деревьев даёт другие вероятности."""
    X, y = small_train
    small = build_pipeline("xgb", "A", {"n_estimators": 10}, cfg).fit(X, y).predict_proba(X)[:, 1]
    large = build_pipeline("xgb", "A", {"n_estimators": 40}, cfg).fit(X, y).predict_proba(X)[:, 1]
    assert not np.array_equal(small, large)
