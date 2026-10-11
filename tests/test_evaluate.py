"""Тесты этапа 6: формула прибыли, подбор порога только по OOF, изоляция теста, число деревьев tuned.

Тестовая часть split читается ровно один раз и только для проверки страховки протокола.
Финальные этапы (таблица, эталонные стратегии, чувствительность, фигуры, parquet) прогоняются
целиком на синтетических метках и вероятностях: обученные модели для них не нужны, но проверка
ловит опечатки в колонках таблиц и обращения к необъявленным именам.
"""
from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.config import load_business_config
from src.evaluate import (
    TABLE_COLUMNS,
    TUNED_LOGREG_FIXED,
    ModelSpec,
    benchmark_table,
    build_sensitivity_table,
    build_specs,
    final_best_keys,
    load_test_frame,
    make_figures,
    oof_predictions,
    predictions_frame,
    print_benchmark_report,
    profit_at_threshold,
    reset_test_reads,
    save_test_artifacts,
    select_threshold,
    tuned_params,
)
from src.pipelines import DUMMY_MODEL
from src.tune import FIXED_LOGREG_PARAMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOST_TREE_PARAM = {"xgb": "n_estimators", "lgbm": "n_estimators", "catboost": "iterations"}
TINY_ROWS = 200
FOLD_COUNT = 3


def _tiny_data(small_train, cfg, n_rows: int = TINY_ROWS) -> tuple:
    X, y = small_train
    sample = X.sample(n=n_rows, random_state=cfg.seed)
    return sample, y.loc[sample.index]


def _fold_data(small_train, cfg, n_negatives: int = 40) -> tuple:
    """Данные для теста OOF: все ушедшие small_train плюс часть оставшихся.

    Гарантии: в каждой train-части стратифицированного фолда оба класса (не меньше 20
    положительных), а число положительных не кратно числу фолдов — иначе train-части совпадали
    бы по составу классов и проверка «модель переобучается на каждом фолде» была бы пустой.
    """
    X, y = small_train
    positives = y[y == 1].index
    if len(positives) % FOLD_COUNT == 0:
        positives = positives[:-1]
    index = positives.append(y[y == 0].sample(n=n_negatives, random_state=cfg.seed).index)
    return X.loc[index], y.loc[index]


def test_profit_manual_example() -> None:
    """Прибыль на ручном примере: один ушедший в контакте и один ложный контакт дают 50."""
    y = np.array([1, 0, 1])
    proba = np.array([0.9, 0.5, 0.1])
    clv, offer_cost, success_rate = 500.0, 50.0, 0.3

    profit = profit_at_threshold(y, proba, 0.5, clv, offer_cost, success_rate)
    print(
        f"y = {y.tolist()}, p = {proba.tolist()}, порог = 0.5: контактов = 2, удержанных = 1, "
        f"прибыль = {profit}"
    )
    assert profit == 50.0
    assert profit_at_threshold(y, proba, 0.1, clv, offer_cost, success_rate) == 150.0
    assert profit_at_threshold(y, proba, 0.95, clv, offer_cost, success_rate) == 0.0


def test_threshold_selection_uses_only_oof() -> None:
    """Порог считается по OOF-меткам и вероятностям; на тестовых данных он был бы другим."""
    argument_names = list(inspect.signature(select_threshold).parameters)
    y_oof, proba_oof = np.array([1, 1, 0]), np.array([0.9, 0.7, 0.2])
    y_test, proba_test = np.array([0, 0, 1]), np.array([0.9, 0.7, 0.2])

    oof_threshold, oof_profit = select_threshold(y_oof, proba_oof, 500.0, 50.0, 0.3)
    test_threshold, test_profit = select_threshold(y_test, proba_test, 500.0, 50.0, 0.3)
    repeated = select_threshold(y_oof, proba_oof, 500.0, 50.0, 0.3)

    print(f"аргументы select_threshold = {argument_names}")
    print(
        f"OOF: порог = {oof_threshold:.2f}, прибыль = {oof_profit:.0f}; если бы порог подбирался "
        f"на тесте: {test_threshold:.2f} (прибыль {test_profit:.0f})"
    )
    assert not any("test" in name for name in argument_names)
    assert oof_threshold == 0.21
    assert oof_profit == 200.0
    assert repeated == (oof_threshold, oof_profit)
    assert test_threshold != oof_threshold


def test_tuned_boosting_tree_count_from_median() -> None:
    """У tuned бустинга число деревьев берётся из median_best_iteration, а не из best_params."""
    entry = {"best_params": {"n_estimators": 600, "max_depth": 3}, "median_best_iteration": 134}
    params = tuned_params("xgb", entry)
    lgbm = tuned_params("lgbm", {"best_params": {"n_estimators": 2000}, "median_best_iteration": 63})
    catboost = tuned_params(
        "catboost", {"best_params": {"iterations": 1000, "depth": 4}, "median_best_iteration": 240}
    )
    logreg = tuned_params("logreg", {"best_params": {"C": 0.5}, "median_best_iteration": None})

    print(f"xgb: best_params = {entry['best_params']}, median = {entry['median_best_iteration']} -> {params}")
    print(f"lgbm: {lgbm}, catboost: {catboost}, logreg: {logreg}")
    assert params["n_estimators"] == 134 and params["max_depth"] == 3
    assert entry["best_params"]["n_estimators"] == 600
    assert lgbm["n_estimators"] == 63
    assert catboost["iterations"] == 240 and catboost["depth"] == 4
    assert logreg == {"C": 0.5, **FIXED_LOGREG_PARAMS}
    assert TUNED_LOGREG_FIXED == FIXED_LOGREG_PARAMS

    with pytest.raises(ValueError):
        tuned_params("xgb", {"best_params": {"n_estimators": 100}, "median_best_iteration": None})


def test_final_configurations_are_wide_and_logreg_a() -> None:
    """Финальные конфигурации — только *_wide у бустингов и logreg_A; в бенчмарке 15 моделей."""
    keys = final_best_keys()
    payload = {
        key: {
            "best_params": {BOOST_TREE_PARAM[key.split("_")[0]]: 100},
            "median_best_iteration": 10,
        }
        for key in keys
        if key != "logreg_A"
    }
    payload["logreg_A"] = {"best_params": {"C": 0.5}, "median_best_iteration": None}

    specs = build_specs(payload)
    tuned_keys = [spec.key for spec in specs if spec.kind == "tuned"]
    print(f"финальные ключи = {keys}")
    print(f"моделей в бенчмарке = {len(specs)}: {[spec.key for spec in specs]}")
    print(f"tuned-модели = {tuned_keys}")

    assert keys == [
        "xgb_A_wide",
        "xgb_B_wide",
        "lgbm_A_wide",
        "lgbm_B_wide",
        "catboost_A_wide",
        "catboost_B_wide",
        "logreg_A",
    ]
    assert len(specs) == 15
    assert tuned_keys == [
        "logreg_tuned",
        "xgb_A_tuned",
        "xgb_B_tuned",
        "lgbm_A_tuned",
        "lgbm_B_tuned",
        "catboost_A_tuned",
        "catboost_B_tuned",
    ]
    assert [spec.key for spec in specs if spec.best_key] == tuned_keys
    assert all(spec.params is None for spec in specs if spec.kind == "default")

    with pytest.raises(KeyError):
        build_specs({key: payload[key] for key in keys[:-1]})


def test_test_split_is_touched_only_by_allowed_modules() -> None:
    """Тестовый split трогают только data.py, evaluate.py (единственное чтение) и explain.py (SHAP)."""
    markers = ("test_idx", "load_test_frame")
    hits = sorted(
        path.name
        for path in (PROJECT_ROOT / "src").glob("*.py")
        if any(marker in path.read_text(encoding="utf-8") for marker in markers)
    )
    print(f"файлы src/ с обращением к тестовому split (маркеры {markers}) = {hits}")
    assert hits == ["data.py", "evaluate.py", "explain.py"]


def test_test_frame_is_read_once(cfg) -> None:
    """Повторное чтение теста за запуск запрещено: счётчик бросает RuntimeError."""
    reset_test_reads()
    test = load_test_frame(cfg)
    print(f"первое чтение теста: строк = {len(test)}, доля оттока = {test[cfg.target].mean():.4f}")

    with pytest.raises(RuntimeError):
        load_test_frame(cfg)
    reset_test_reads()


def test_oof_predictions_refit_on_each_fold(small_train, cfg) -> None:
    """OOF собирается покомпонентно: dummy повторяет долю оттока train-части каждого фолда."""
    X, y = _fold_data(small_train, cfg)
    run_cfg = replace(cfg, n_jobs=1)
    y_values = y.to_numpy()
    folds = list(StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=42).split(X, y))
    dummy = ModelSpec("dummy", "dummy", "A", "default", None, None, "DummyClassifier")
    logreg = ModelSpec("logreg_default", "logreg", "A", "default", None, None, "LogisticRegression")

    proba_dummy, seconds = oof_predictions(dummy, X, y, folds, run_cfg)
    proba_logreg, _ = oof_predictions(logreg, X, y, folds, run_cfg)
    fold_priors = [float(y_values[train_idx].mean()) for train_idx, _ in folds]
    train_positives = [int(y_values[train_idx].sum()) for train_idx, _ in folds]

    print(f"строк = {len(X)}, положительных = {int(y_values.sum())}, фолдов = {len(folds)}")
    print(f"положительных в train-частях фолдов = {train_positives}")
    print(f"доля оттока по train-частям фолдов = {fold_priors}")
    print(
        f"OOF logreg: min = {proba_logreg.min():.4f}, max = {proba_logreg.max():.4f}, "
        f"уникальных значений = {len(np.unique(proba_logreg))}, время dummy = {np.mean(seconds):.4f} с/фолд"
    )

    assert proba_dummy.shape == (len(X),) and proba_logreg.shape == (len(X),)
    assert len(X) >= 60 and int(y_values.sum()) >= 20
    assert min(train_positives) >= 20
    assert len({round(prior, 6) for prior in fold_priors}) > 1
    for train_idx, valid_idx in folds:
        assert np.allclose(proba_dummy[valid_idx], y_values[train_idx].mean())
    assert np.isfinite(proba_logreg).all()
    assert len(np.unique(proba_logreg)) > 10


def test_oof_raises_without_positive_class_in_fold(small_train, cfg) -> None:
    """Одноклассовая train-часть фолда: вместо нулей выбрасывается ValueError."""
    X, y = _tiny_data(small_train, cfg)
    run_cfg = replace(cfg, n_jobs=1)
    y_values = y.to_numpy()
    positives = np.flatnonzero(y_values == 1)
    negatives = np.flatnonzero(y_values == 0)
    train_idx = negatives[:80]
    valid_idx = np.concatenate([negatives[80:100], positives[:20]])
    dummy = ModelSpec("dummy", "dummy", "A", "default", None, None, "DummyClassifier")

    print(
        f"train-часть фолда: строк = {len(train_idx)}, положительных = {int(y_values[train_idx].sum())}; "
        f"валидация: строк = {len(valid_idx)}"
    )
    with pytest.raises(ValueError, match="положительного класса"):
        oof_predictions(dummy, X, y, [(train_idx, valid_idx)], run_cfg)


def test_final_stage_artifacts_on_synthetic_probabilities(tmp_path, cfg) -> None:
    """Финальная часть без обучения: таблица, эталоны, чувствительность, фигуры и parquet в tmp_path.

    Проверяются ровно те функции, которые вызывает run_benchmark после предсказаний теста, поэтому
    тест падает на опечатке в колонке таблицы или на обращении к необъявленному имени.
    """
    business = load_business_config()
    rng = np.random.default_rng(cfg.seed)
    size = 300
    y_test = pd.Series((rng.random(size) < 0.30).astype(int), name=cfg.target)
    signal = 0.25 * y_test.to_numpy() + rng.normal(0.0, 0.10, size)
    test_proba = {
        DUMMY_MODEL: np.full(size, float(y_test.mean())),
        "model_a_tuned": np.clip(signal + 0.05, 0.0, 1.0),
        "model_b_default": np.clip(signal, 0.0, 1.0),
        "model_c_default": np.clip(signal - 0.05, 0.0, 1.0),
    }
    specs = [
        ModelSpec(DUMMY_MODEL, DUMMY_MODEL, "A", "default", None, None, "DummyClassifier"),
        ModelSpec("model_a_tuned", "logreg", "A", "tuned", None, None, "модель A tuned"),
        ModelSpec("model_b_default", "logreg", "A", "default", None, None, "модель B default"),
        ModelSpec("model_c_default", "xgb", "A", "default", None, None, "модель C default"),
    ]
    thresholds = {spec.key: {"threshold": 0.40} for spec in specs if spec.uses_threshold}
    oof_proba = {key: np.clip(proba - 0.02, 0.0, 1.0) for key, proba in test_proba.items()}
    y_oof = y_test.to_numpy()
    oof_seconds = {spec.key: [0.10, 0.12, 0.11] for spec in specs}
    fit_seconds = {spec.key: 0.5 + index for index, spec in enumerate(specs)}
    infer_seconds = {spec.key: 0.001 * (index + 1) for index, spec in enumerate(specs)}
    rates = [0.1, 0.3, 0.4]
    clv = business.customer_lifetime_value
    cost = business.retention_offer_cost
    rate = business.retention_success_rate

    table = benchmark_table(
        specs, y_test, test_proba, thresholds, oof_seconds, fit_seconds, infer_seconds, 1200, business
    )
    sensitivity = build_sensitivity_table(specs, y_oof, oof_proba, y_test, test_proba, business, rates)
    by_key = table.set_index("key")
    expected_profit = profit_at_threshold(y_test, test_proba["model_a_tuned"], 0.40, clv, cost, rate)
    expected_contact_all = float(y_test.sum()) * clv * rate - len(y_test) * cost
    threshold_oof, oof_profit = select_threshold(y_oof, oof_proba["model_a_tuned"], clv, cost, 0.3)
    row = sensitivity[(sensitivity["key"] == "model_a_tuned") & (sensitivity["success_rate"] == 0.3)].iloc[0]

    print(
        f"таблица: строк = {len(table)}, колонок = {table.shape[1]}, доля оттока = {y_test.mean():.4f}; "
        f"чувствительность: строк = {len(sensitivity)}"
    )
    print(
        f"прибыль model_a_tuned = {expected_profit:,.0f}, эталон «контактировать всех» = "
        f"{expected_contact_all:,.0f}, порог по OOF = {threshold_oof:.2f}"
    )

    assert tuple(table.columns) == TABLE_COLUMNS
    assert len(table) == len(specs) + 2
    assert float(by_key.loc["model_a_tuned", "profit_at_threshold"]) == pytest.approx(round(expected_profit, 2))
    assert float(by_key.loc["reference_contact_none", "profit_at_threshold"]) == 0.0
    assert float(by_key.loc["reference_contact_all", "profit_at_threshold"]) == pytest.approx(
        round(expected_contact_all, 2)
    )
    assert int(by_key.loc["reference_contact_all", "contacts_at_threshold"]) == len(y_test)
    assert pd.isna(by_key.loc[DUMMY_MODEL, "threshold"]) and pd.isna(by_key.loc[DUMMY_MODEL, "profit_at_threshold"])

    expected_keys = {spec.key for spec in specs if spec.uses_threshold}
    assert set(sensitivity["key"]) == expected_keys | {"reference_contact_none", "reference_contact_all"}
    assert len(sensitivity) == len(expected_keys) * len(rates) + 2 * len(rates)
    assert float(row["threshold_oof"]) == pytest.approx(threshold_oof)
    assert float(row["oof_profit"]) == pytest.approx(round(oof_profit, 2))
    assert int(row["n_contacts"]) == int((test_proba["model_a_tuned"] >= threshold_oof).sum())

    test_frame = predictions_frame(y_test, test_proba)
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    artifact_paths = save_test_artifacts(table, test_frame, sensitivity, tables_dir)
    figure_paths = make_figures(
        [spec for spec in specs if spec.uses_threshold],
        oof_proba,
        test_proba,
        y_oof,
        y_test.to_numpy(),
        business,
        figures_dir,
    )
    print_benchmark_report(table, sensitivity, {}, specs)

    assert [path.name for path in artifact_paths] == [
        "test_benchmark.csv",
        "test_predictions.parquet",
        "business_sensitivity.csv",
    ]
    assert [path.name for path in figure_paths] == [
        "test_calibration.png",
        "test_profit_curves.png",
        "test_pr_curves.png",
    ]
    for path in (*artifact_paths, *figure_paths):
        assert path.exists() and path.stat().st_size > 0
    print(f"артефакты в tmp_path: {[path.name for path in (*artifact_paths, *figure_paths)]}")

    restored_table = pd.read_csv(tables_dir / "test_benchmark.csv")
    restored_predictions = pd.read_parquet(tables_dir / "test_predictions.parquet")
    restored_sensitivity = pd.read_csv(tables_dir / "business_sensitivity.csv")
    assert tuple(restored_table.columns) == TABLE_COLUMNS
    assert len(restored_table) == len(table)
    assert list(restored_predictions.columns) == list(test_frame.columns)
    assert np.allclose(restored_predictions["y_true"].to_numpy(), y_test.to_numpy())
    assert np.allclose(restored_predictions[f"p_{DUMMY_MODEL}"].to_numpy(), test_proba[DUMMY_MODEL])
    assert len(restored_sensitivity) == len(sensitivity)
