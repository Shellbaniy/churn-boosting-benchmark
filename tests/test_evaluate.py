"""Тесты этапа 6: формула прибыли, подбор порога только по OOF, изоляция теста, число деревьев tuned.

Тестовая часть split читается ровно один раз и только для проверки страховки протокола.
"""
from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from src.evaluate import (
    TUNED_LOGREG_FIXED,
    ModelSpec,
    build_specs,
    final_best_keys,
    load_test_frame,
    oof_predictions,
    profit_at_threshold,
    reset_test_reads,
    select_threshold,
    tuned_params,
)
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


def test_test_idx_referenced_only_in_data_and_evaluate() -> None:
    """Ссылка на test_idx встречается в src/ только у data.py и evaluate.py (аудит утечки)."""
    hits = sorted(
        path.name
        for path in (PROJECT_ROOT / "src").glob("*.py")
        if "test_idx" in path.read_text(encoding="utf-8")
    )
    print(f"файлы src/ со ссылкой на test_idx = {hits}")
    assert hits == ["data.py", "evaluate.py"]


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
