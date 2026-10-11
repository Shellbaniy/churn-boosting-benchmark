"""Мини-тесты этапа 7: формула corrected resampled t-test, Holm, McNemar, парность бутстрэпа.

Все расчёты проверяются на ручных примерах и синтетических вероятностях: обученные модели и
реальные данные нужны только там, где проверяется общая схема разбиений (`small_train`).
Тестовая часть split здесь не читается: тестовые числа этапа 7 берутся из артефакта этапа 6.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from scipy import stats as sps

from src.config import load_business_config
from src.stats import (
    FAMILY_H1,
    FAMILY_H2,
    FAMILY_H3,
    Pair,
    TunedModel,
    bootstrap_diff_frame,
    bootstrap_distributions,
    corrected_resampled_ttest,
    cv_score_frame,
    cv_scores,
    holm_correction,
    mcnemar_exact_p,
    mcnemar_frame,
    preregistered_pairs,
    repeated_folds,
    run_stats,
    tuned_models,
)
from src.utils import ensure_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_REPLICATES = 200
LOCAL_FOLDS = 3
LOCAL_REPEATS = 2


def test_corrected_resampled_ttest_matches_manual_example() -> None:
    """SE и t считаются по формуле Nadeau & Bengio, поправка n_test/n_train гасит значимость."""
    differences = np.array([0.010, 0.020, 0.030, 0.000])
    ratio = 0.25
    result = corrected_resampled_ttest(differences, ratio_test_train=ratio)

    variance = float(differences.var(ddof=1))
    expected_se = float(np.sqrt((1.0 / differences.size + ratio) * variance))
    expected_t = float(differences.mean() / expected_se)
    expected_p = float(2.0 * sps.t.sf(abs(expected_t), differences.size - 1))
    plain = corrected_resampled_ttest(differences, ratio_test_train=0.0)

    print(
        f"различия = {differences.tolist()}, n_test/n_train = {ratio}: mean = {result['mean_diff']:.4f}, "
        f"SE = {result['se']:.6f}, t = {result['t']:.4f}, df = {result['df']:.0f}, p = {result['p']:.4f}"
    )
    print(
        f"без поправки на перекрытие фолдов: SE = {plain['se']:.6f}, t = {plain['t']:.4f}, "
        f"p = {plain['p']:.4f}"
    )

    assert result["n_folds"] == differences.size and result["df"] == differences.size - 1
    assert result["se"] == pytest.approx(expected_se)
    assert result["t"] == pytest.approx(expected_t)
    assert result["p"] == pytest.approx(expected_p)
    assert result["d_z"] == pytest.approx(float(differences.mean() / differences.std(ddof=1)))
    assert result["ci_low"] < result["mean_diff"] < result["ci_high"]
    assert plain["se"] < result["se"] and plain["p"] < result["p"]

    with pytest.raises(ValueError):
        corrected_resampled_ttest(np.array([0.01]), 0.25)


def test_holm_correction_matches_manual_example() -> None:
    """Holm: три p-value [0.01, 0.04, 0.03] превращаются в [0.03, 0.06, 0.06] в исходном порядке."""
    adjusted = holm_correction([0.01, 0.04, 0.03])
    print(f"p = [0.01, 0.04, 0.03] → p_holm = {np.round(adjusted, 6).tolist()}")

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert holm_correction([]).size == 0
    assert holm_correction([0.5]).tolist() == [0.5]
    with pytest.raises(ValueError):
        holm_correction([1.5])


def test_mcnemar_exact_matches_binomial_example() -> None:
    """Точный McNemar — биномиальный тест: b = 8, c = 2 дают p = 2 × P(X ≥ 8) = 0.109375."""
    p_value = mcnemar_exact_p(8, 2)
    expected = 2.0 * (comb(10, 8) + comb(10, 9) + comb(10, 10)) / 2**10
    print(f"b = 8, c = 2: p = {p_value:.6f} (вручную 2 × 56 / 1024 = {expected:.6f})")

    assert p_value == pytest.approx(0.109375)
    assert p_value == pytest.approx(expected)
    assert mcnemar_exact_p(2, 8) == pytest.approx(p_value)
    assert mcnemar_exact_p(5, 5) == 1.0
    assert mcnemar_exact_p(0, 0) == 1.0

    y_true = np.array([1.0, 1.0, 0.0, 0.0])
    proba = {"a": np.array([0.9, 0.1, 0.2, 0.8]), "b": np.array([0.9, 0.6, 0.2, 0.8])}
    thresholds = {"a": 0.5, "b": 0.5}
    models = [TunedModel("a", "logreg", "A"), TunedModel("b", "xgb", "A")]
    table = mcnemar_frame(y_true, proba, thresholds, [Pair(FAMILY_H1, "a", "b")], models)
    row = table.iloc[0]
    print(
        f"на четырёх наблюдениях: b (A права) = {int(row['b_a_right'])}, c (B права) = "
        f"{int(row['c_b_right'])}, p = {row['p']:.4f}"
    )

    assert int(row["b_a_right"]) == 0 and int(row["c_b_right"]) == 1
    assert int(row["n_discordant"]) == 1
    assert float(row["p"]) == pytest.approx(1.0)
    assert float(row["p_holm"]) == pytest.approx(1.0)


def test_bootstrap_is_paired_seeded_and_zero_for_same_ranking(cfg) -> None:
    """Индексы строк общие для всех моделей: при одинаковом ранжировании разность ровно нулевая."""
    business = load_business_config()
    y_true = np.concatenate([np.ones(20), np.zeros(60)])
    rng = np.random.default_rng(cfg.seed)
    base = np.concatenate([rng.uniform(0.55, 0.95, 20), rng.uniform(0.05, 0.45, 60)])
    proba = {"a": base, "b": base * 0.5}
    thresholds = {"a": 0.5, "b": 0.25}
    keys = ["a", "b"]

    first, meta_first = bootstrap_distributions(
        y_true, proba, thresholds, business, keys, n_bootstrap=BOOTSTRAP_REPLICATES, seed=cfg.seed
    )
    second, meta_second = bootstrap_distributions(
        y_true, proba, thresholds, business, keys, n_bootstrap=BOOTSTRAP_REPLICATES, seed=cfg.seed
    )
    models = [TunedModel("a", "logreg", "A"), TunedModel("b", "logreg", "A")]
    diff_table = bootstrap_diff_frame(first, [Pair(FAMILY_H1, "a", "b")], models)
    roc_row = diff_table.loc[diff_table["metric"] == "roc_auc"].iloc[0]

    print(
        f"строк в выборке = {meta_first['n_test']}, бутстрэп-выборок = {meta_first['n_bootstrap']}, "
        f"пропущено = {meta_first['skipped']}, seed = {meta_first['seed']}"
    )
    print(
        f"равное ранжирование: разность ROC-AUC = {roc_row['diff_mean']:.2e}, "
        f"CI = [{roc_row['ci_low']:.2e}, {roc_row['ci_high']:.2e}], p = {roc_row['p']:.4f}"
    )

    assert meta_first["n_test"] == y_true.size
    assert meta_first["seed"] == cfg.seed == meta_second["seed"]
    assert meta_first["n_bootstrap"] == BOOTSTRAP_REPLICATES - meta_first["skipped"]
    assert meta_first["n_bootstrap"] >= BOOTSTRAP_REPLICATES - 1
    assert np.allclose(first["a"]["roc_auc"], second["a"]["roc_auc"])
    for metric in ("roc_auc", "pr_auc", "profit_per_customer"):
        assert np.allclose(first["a"][metric], first["b"][metric])
        subset = diff_table.loc[diff_table["metric"] == metric]
        assert np.allclose(subset["diff_mean"].to_numpy(), 0.0)
        assert np.allclose(subset["ci_low"].to_numpy(), 0.0)
        assert np.allclose(subset["ci_high"].to_numpy(), 0.0)
        assert (subset["p"].to_numpy() == 1.0).all()
    assert float(roc_row["p_holm"]) == 1.0


def test_preregistered_pairs_cover_three_families() -> None:
    """15 пар: H1 — 6 бустингов против logreg_tuned, H2 — 3 пары A/B, H3 — 6 пар алгоритмов."""
    models = tuned_models()
    pairs = preregistered_pairs(models)
    keys = [model.key for model in models]
    families = Counter(pair.family for pair in pairs)
    logreg = next(model for model in models if model.model_name == "logreg")
    xgb = next(model for model in models if model.key == "xgb_A_wide")

    print(f"модели: {keys}")
    print(f"пары по семействам: {dict(families)}")

    assert len(keys) == 7 and keys[-1] == "logreg_A"
    assert set(keys) == {
        "xgb_A_wide",
        "xgb_B_wide",
        "lgbm_A_wide",
        "lgbm_B_wide",
        "catboost_A_wide",
        "catboost_B_wide",
        "logreg_A",
    }
    assert len(pairs) == len({(pair.a, pair.b) for pair in pairs}) == 15
    assert families[FAMILY_H1] == 6 and families[FAMILY_H2] == 3 and families[FAMILY_H3] == 6
    assert all(pair.a != pair.b for pair in pairs)
    assert {pair.b for pair in pairs if pair.family == FAMILY_H1} == {logreg.key}
    assert logreg.evaluate_key == "logreg_tuned"
    assert xgb.evaluate_key == "xgb_A_tuned"
    assert [(model.key, model.evaluate_key) for model in models] == [
        ("xgb_A_wide", "xgb_A_tuned"),
        ("xgb_B_wide", "xgb_B_tuned"),
        ("lgbm_A_wide", "lgbm_A_tuned"),
        ("lgbm_B_wide", "lgbm_B_tuned"),
        ("catboost_A_wide", "catboost_A_tuned"),
        ("catboost_B_wide", "catboost_B_tuned"),
        ("logreg_A", "logreg_tuned"),
    ]
    assert logreg.label.endswith("A")


def test_all_models_get_the_same_splits(small_train, cfg, monkeypatch) -> None:
    """cv_score_frame обучает каждую модель на одних и тех же индексах всех разбиений."""
    X, y = small_train
    local_cfg = replace(cfg, cv_splits=LOCAL_FOLDS, cv_repeats=LOCAL_REPEATS)
    models = [
        TunedModel("logreg_A", "logreg", "A"),
        TunedModel("xgb_A_wide", "xgb", "A"),
        TunedModel("catboost_A_wide", "catboost", "A"),
    ]
    payload = {
        "logreg_A": {"best_params": {"C": 0.5}, "median_best_iteration": None},
        "xgb_A_wide": {"best_params": {"max_depth": 2}, "median_best_iteration": 15},
        "catboost_A_wide": {"best_params": {"depth": 2}, "median_best_iteration": 15},
    }
    calls: dict[str, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}

    class RecordingPipeline:
        """Заглушка Pipeline: запоминает индексы строк, на которых её обучали и предсказывали."""

        classes_ = [0, 1]

        def __init__(self, name: str) -> None:
            self.name = name
            self.train_index: tuple[int, ...] = ()

        def fit(self, X_part, y_part):  # noqa: N803
            self.train_index = tuple(X_part.index)
            return self

        def predict_proba(self, X_part):  # noqa: N803
            calls.setdefault(self.name, []).append((self.train_index, tuple(X_part.index)))
            proba = np.full(len(X_part), 0.5)
            return np.column_stack([1.0 - proba, proba])

    monkeypatch.setattr(
        "src.stats.build_pipeline",
        lambda model_name, variant, params, part: RecordingPipeline(model_name),
    )

    folds_first = repeated_folds(X, y, local_cfg)
    folds_second = repeated_folds(X, y, local_cfg)
    frame = cv_score_frame(models, X, y, local_cfg, folds_first, payload)
    reference = calls["logreg"]

    print(f"разбиений = {len(folds_first)} (cv_splits = {LOCAL_FOLDS} × cv_repeats = {LOCAL_REPEATS})")
    print(f"строк в таблице CV = {len(frame)}, моделей в вызовах = {len(calls)}")

    assert len(folds_first) == LOCAL_FOLDS * LOCAL_REPEATS
    assert all(
        np.array_equal(left[0], right[0]) and np.array_equal(left[1], right[1])
        for left, right in zip(folds_first, folds_second)
    )
    for repeat in range(LOCAL_REPEATS):
        covered = np.sort(
            np.concatenate([folds_first[repeat * LOCAL_FOLDS + fold][1] for fold in range(LOCAL_FOLDS)])
        )
        assert np.array_equal(covered, np.arange(len(X)))

    assert len(frame) == len(models) * len(folds_first)
    assert set(frame["model"]) == {model.key for model in models}
    assert frame["roc_auc"].nunique() == 1 and frame["pr_auc"].between(0.0, 1.0).all()
    for name in ("xgb", "catboost"):
        assert calls[name] == reference
    for position, (train_index, valid_index) in enumerate(reference):
        fold_train, fold_valid = folds_first[position]
        assert len(train_index) + len(valid_index) == len(X)
        assert set(train_index).isdisjoint(valid_index)
        assert np.array_equal(np.array(valid_index), X.index[fold_valid].to_numpy())
        assert np.array_equal(np.array(train_index), X.index[fold_train].to_numpy())


def test_stats_end_to_end_on_synthetic_predictions(small_train, cfg, tmp_path) -> None:
    """Все три части на синтетике: таблицы, фигуры и кэш CV в tmp_path."""
    X, y = small_train
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    local_cfg = replace(
        cfg,
        cv_splits=LOCAL_FOLDS,
        cv_repeats=LOCAL_REPEATS,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    )
    models = [
        TunedModel("logreg_A", "logreg", "A"),
        TunedModel("xgb_A_wide", "xgb", "A"),
        TunedModel("lgbm_A_wide", "lgbm", "A"),
    ]
    pairs = [
        Pair(FAMILY_H1, "xgb_A_wide", "logreg_A"),
        Pair(FAMILY_H1, "lgbm_A_wide", "logreg_A"),
        Pair(FAMILY_H3, "xgb_A_wide", "lgbm_A_wide"),
    ]
    payload = {
        "logreg_A": {"best_params": {"C": 0.5}, "median_best_iteration": None},
        "xgb_A_wide": {"best_params": {"max_depth": 2, "learning_rate": 0.3}, "median_best_iteration": 15},
        "lgbm_A_wide": {
            "best_params": {"num_leaves": 4, "learning_rate": 0.3, "min_child_samples": 5},
            "median_best_iteration": 15,
        },
    }
    n_test = 300
    y_test = np.concatenate([np.ones(90), np.zeros(n_test - 90)])
    rng = np.random.default_rng(cfg.seed + 1)
    signal = np.concatenate([rng.uniform(0.4, 0.9, 90), rng.uniform(0.05, 0.6, n_test - 90)])
    predictions = pd.DataFrame({"y_true": y_test})
    for offset, model in enumerate(models):
        noise = rng.normal(0.0, 0.02 * (1 + offset), size=n_test)
        predictions[f"p_{model.evaluate_key}"] = np.clip(signal + noise, 0.001, 0.999)
    ensure_dir(tables_dir)
    predictions.to_parquet(tables_dir / "test_predictions.parquet", index=False)
    thresholds = {
        "thresholds": {model.evaluate_key: {"threshold": 0.4, "profit": 0.0} for model in models}
    }
    (tables_dir / "thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False), encoding="utf-8")

    paths = run_stats(
        cfg=local_cfg,
        n_bootstrap=BOOTSTRAP_REPLICATES,
        models=models,
        pairs=pairs,
        payload=payload,
        train=(X, y),
    )
    ttest_table = pd.read_csv(paths["ttest"])
    ci_table = pd.read_csv(paths["bootstrap_ci"])
    diff_table = pd.read_csv(paths["bootstrap_diff"])
    mcnemar_table = pd.read_csv(paths["mcnemar"])

    print(f"артефакты: {sorted(path.name for path in paths.values())}")

    assert set(paths) == {
        "cv_scores",
        "ttest",
        "bootstrap_ci",
        "bootstrap_diff",
        "mcnemar",
        "cv_boxplot",
        "auc_ci",
        "pvalue_heatmap",
    }
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0
    assert len(ttest_table) == len(pairs) * 2 and set(ttest_table["metric"]) == {"roc_auc", "pr_auc"}
    assert len(ci_table) == len(models) * 3 and len(diff_table) == len(pairs) * 3
    assert len(mcnemar_table) == len(pairs)
    assert set(ttest_table["evaluate_key_b"]) <= {model.evaluate_key for model in models}
    assert set(ci_table["evaluate_key"]) == {model.evaluate_key for model in models}
    assert set(diff_table["evaluate_key_a"]) <= {model.evaluate_key for model in models}
    assert set(mcnemar_table["pair"]) == {pair.name for pair in pairs}
    assert np.isfinite(ttest_table["p"]).all() and np.isfinite(diff_table["p"]).all()
    assert (ttest_table["p_holm"] >= ttest_table["p"] - 1e-12).all()
    assert (diff_table["p_holm"] >= diff_table["p"] - 1e-12).all()
    assert (diff_table["p"] >= 1.0 / BOOTSTRAP_REPLICATES - 1e-12).all()
    assert mcnemar_table["n_discordant"].between(0, n_test).all()

    cached, ratio_test_train = cv_scores(models, X, y, local_cfg, payload, force=False)
    forced, forced_ratio = cv_scores(models, X, y, local_cfg, payload, force=True)
    stored = pd.read_csv(paths["cv_scores"])

    print(
        f"CV: строк = {len(stored)}, n_test/n_train = {ratio_test_train:.4f} "
        f"(после --force {forced_ratio:.4f}), среднее ROC-AUC = {stored['roc_auc'].mean():.4f}"
    )

    assert len(stored) == len(models) * LOCAL_FOLDS * LOCAL_REPEATS
    assert len(cached) == len(stored) == len(forced)
    assert ratio_test_train == pytest.approx(1.0 / (LOCAL_FOLDS - 1), abs=0.01)
    assert np.allclose(cached["roc_auc"].to_numpy(), forced["roc_auc"].to_numpy())


def test_stats_source_does_not_read_test_split_again() -> None:
    """Этап 7 не обращается к тестовой части split: только артефакты этапа 6."""
    source = (PROJECT_ROOT / "src" / "stats.py").read_text(encoding="utf-8")

    print(f"строк в src/stats.py = {source.count(chr(10)) + 1}")
    print(f"упоминания артефактов этапа 6: test_predictions.parquet = {'test_predictions.parquet' in source}")

    assert "test_idx" not in source
    assert "load_test_frame" not in source
    assert "reset_test_reads" not in source
    assert "test_predictions.parquet" in source


def test_evaluate_keys_match_stage6_artifacts() -> None:
    """Отображение best_params → этап 6 сверяется с реальными артефактами.

    Читаются только метаданные parquet (список колонок, без данных) и ключи thresholds.json:
    данные теста при этом не загружаются.
    """
    tables_dir = PROJECT_ROOT / "reports" / "tables"
    predictions_path = tables_dir / "test_predictions.parquet"
    thresholds_path = tables_dir / "thresholds.json"
    if not predictions_path.exists() or not thresholds_path.exists():
        pytest.skip("нет артефактов этапа 6 (test_predictions.parquet / thresholds.json): выполните make benchmark")

    columns = list(pq.ParquetFile(predictions_path).schema.names)
    entries = json.loads(thresholds_path.read_text(encoding="utf-8")).get("thresholds", {})
    models = tuned_models()
    missing_columns = [model.evaluate_key for model in models if f"p_{model.evaluate_key}" not in columns]
    missing_thresholds = [model.evaluate_key for model in models if model.evaluate_key not in entries]

    print(f"колонок в test_predictions.parquet = {len(columns)}: {columns}")
    print(f"записей в thresholds.json = {len(entries)}: {sorted(entries)}")
    print(f"проверено моделей = {len(models)}; нет колонок: {missing_columns}; нет порогов: {missing_thresholds}")

    assert "y_true" in columns
    assert len(models) == 7
    assert missing_columns == []
    assert missing_thresholds == []
