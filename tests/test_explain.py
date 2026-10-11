"""Мини-тесты этапа 8: аддитивность SHAP, агрегация OHE-столбцов, Спирмен и состав признаков.

SHAP проверяется на компактной подвыборке train с быстрым xgb (глубина 1, 20 деревьев): тестовая
часть split здесь не читается, все проверки идут только на train-подвыборке.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from src.evaluate import ModelSpec
from src.explain import (
    EXPECTED_FEATURES,
    aggregate_values,
    explain_model,
    importance_frame,
    rank_correlation,
    transform_matrix,
)
from src.pipelines import build_pipeline, categorical_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINI_ROWS = 150
MINI_PARAMS = {"max_depth": 1, "n_estimators": 20, "learning_rate": 0.3}
MINI_SPEC = ModelSpec(
    key="xgb_A_tuned",
    model_name="xgb",
    variant="A",
    kind="tuned",
    params=None,
    best_key=None,
    label="XGBoost A tuned (мини-тест)",
)


def fitted_xgb(small_train, cfg):
    """xgb варианта A на подвыборке train: возвращает пайплайн, данные и SHAP-результат."""
    X, y = small_train
    sample = X.sample(n=MINI_ROWS, random_state=cfg.seed)
    X_mini = sample
    pipeline = build_pipeline("xgb", "A", MINI_PARAMS, cfg)
    pipeline.fit(X_mini, y.loc[sample.index])
    features = list(X_mini.columns)
    result = explain_model(MINI_SPEC, pipeline, X_mini, features)
    return pipeline, X_mini, features, result


def test_xgb_shap_is_additive_with_raw_margin(small_train, cfg) -> None:
    """sum(SHAP) + базовое значение равно сырому выходу xgb (лог-шансы) на мини-данных."""
    pipeline, X_mini, features, result = fitted_xgb(small_train, cfg)
    matrix, columns = transform_matrix(MINI_SPEC, pipeline, X_mini)
    booster = pipeline.named_steps["model"].get_booster()
    margin = np.asarray(booster.predict(xgb.DMatrix(matrix), output_margin=True), dtype=float).ravel()
    reconstructed = result.raw_values.sum(axis=1) + result.base_value
    error = float(np.max(np.abs(reconstructed - margin)))

    print(
        f"строк = {X_mini.shape[0]}, столбцов матрицы = {len(columns)}, "
        f"базовое значение = {result.base_value:.4f}, макс. |sum(SHAP) + base - margin| = {error:.2e}"
    )

    assert result.raw_values.shape == (X_mini.shape[0], len(columns))
    assert np.allclose(reconstructed, margin, atol=1e-4)
    assert error < 1e-4


def test_ohe_aggregation_preserves_row_sum(small_train, cfg) -> None:
    """Сумма SHAP по строке не меняется при агрегации OHE-столбцов в исходные признаки."""
    pipeline, X_mini, features, result = fitted_xgb(small_train, cfg)
    aggregated = aggregate_values(
        result.raw_values, result.raw_features, categorical_columns(X_mini), features
    )
    raw_sums = result.raw_values.sum(axis=1)
    max_gap = float(np.max(np.abs(aggregated.sum(axis=1) - raw_sums)))

    print(
        f"столбцов матрицы = {result.raw_values.shape[1]}, исходных признаков = {result.values.shape[1]}, "
        f"макс. расхождение сумм по строке = {max_gap:.2e}"
    )

    assert result.raw_values.shape[1] > EXPECTED_FEATURES
    assert np.allclose(aggregated, result.values)
    assert np.allclose(result.values.sum(axis=1), raw_sums, atol=1e-10)
    assert max_gap < 1e-10


def test_rank_correlation_endpoints() -> None:
    """Спирмен: одинаковые ранги дают ρ = 1, обратные — ρ = -1."""
    ranks = np.arange(1, EXPECTED_FEATURES + 1, dtype=float)
    same_rho, same_p = rank_correlation(ranks, ranks)
    reversed_rho, reversed_p = rank_correlation(ranks, ranks[::-1])

    print(
        f"одинаковые ранги: ρ = {same_rho:.4f} (p = {same_p:.4f}), "
        f"обратные ранги: ρ = {reversed_rho:.4f} (p = {reversed_p:.4f})"
    )

    assert same_rho == pytest.approx(1.0) and same_p == pytest.approx(0.0)
    assert reversed_rho == pytest.approx(-1.0) and reversed_p == pytest.approx(0.0)


def test_importance_table_covers_all_source_features(small_train, cfg) -> None:
    """В таблице важности присутствуют все 19 исходных признаков с рангами 1..19."""
    pipeline, X_mini, features, result = fitted_xgb(small_train, cfg)
    importance = importance_frame([result], features)

    print(
        f"строк в таблице важности = {len(importance)}, признаков в данных = {len(X_mini.columns)}, "
        f"первый по важности = "
        f"{importance.loc[importance['rank'] == 1, 'feature'].iloc[0]}"
    )

    assert len(features) == EXPECTED_FEATURES
    assert len(importance) == EXPECTED_FEATURES
    assert set(importance["feature"]) == set(X_mini.columns)
    assert sorted(importance["rank"].tolist()) == list(range(1, EXPECTED_FEATURES + 1))
    assert bool((importance["mean_abs_shap"] >= 0.0).all())


def test_explain_source_does_not_touch_split_indices() -> None:
    """explain.py читает тест через load_test_frame из evaluate.py и не обращается к test_idx."""
    source = (PROJECT_ROOT / "src" / "explain.py").read_text(encoding="utf-8")

    print(f"строк в src/explain.py = {source.count(chr(10)) + 1}")

    assert "test_idx" not in source
    assert "load_test_frame" in source
