"""Тесты тюнинга (этап 5): протокол разбиений, диапазон objective, детерминизм trial.

Тестовая часть split не используется: все тесты работают только с train.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import optuna
from optuna.trial import TrialState

from src.pipelines import make_folds
from src.tune import inner_split, make_objective, normalize_tag, parse_args, run_study
from src.utils import read_json

# Мало деревьев и короткое терпение: тесты должны идти секунды, а не минуты.
TINY_SPACE = {"n_estimators": {"type": "int", "low": 10, "high": 20}}
TINY_ES_ROUNDS = 5
TINY_ROWS = 200


def _tiny_data(train_frame, cfg, n_rows: int = TINY_ROWS) -> tuple:
    X, y = train_frame
    sample = X.sample(n=n_rows, random_state=cfg.seed)
    return sample, y.loc[sample.index]


def test_inner_valid_does_not_overlap_outer_valid(train_frame, cfg) -> None:
    """inner-val строится из train-части фолда и не пересекается с валидацией этого фолда."""
    X, y = train_frame
    outer_train, outer_valid = make_folds(X, y, cfg)[0]

    inner_train, inner_valid = inner_split(y.iloc[outer_train], cfg)
    inner_train_idx = np.asarray(outer_train)[inner_train]
    inner_valid_idx = np.asarray(outer_train)[inner_valid]

    print(
        f"фолд 0: train = {len(outer_train)}, valid = {len(outer_valid)}; "
        f"inner-train = {len(inner_train_idx)}, inner-val = {len(inner_valid_idx)}, "
        f"общих индексов inner-val и valid = {np.intersect1d(inner_valid_idx, outer_valid).size}"
    )
    assert np.intersect1d(inner_valid_idx, outer_valid).size == 0
    assert np.intersect1d(inner_train_idx, inner_valid_idx).size == 0
    assert np.array_equal(np.sort(np.concatenate([inner_train_idx, inner_valid_idx])), np.sort(outer_train))


def test_objective_value_is_roc_auc_in_range(small_train, cfg) -> None:
    """Objective на мини-выборке (200 строк, 1 trial, 2 фолда) возвращает ROC-AUC из [0.5, 1]."""
    X, y = _tiny_data(small_train, cfg)
    folds = make_folds(X, y, replace(cfg, cv_splits=2))
    objective = make_objective("xgb", "A", X, y, folds, replace(cfg, n_jobs=1), TINY_SPACE, TINY_ES_ROUNDS)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    value = float(study.best_value)

    print(
        f"objective: строк = {len(X)}, фолдов = {len(folds)}, деревьев = {study.best_params['n_estimators']}, "
        f"ROC-AUC = {value:.4f}, состояние trial = {study.best_trial.state.name}"
    )
    assert study.best_trial.state == TrialState.COMPLETE
    assert 0.5 <= value <= 1.0


def test_same_seed_gives_same_trial_value(small_train, cfg, tmp_path) -> None:
    """Одинаковый seed (sampler, фолды, inner-split) даёт одинаковое значение trial."""
    X, y = _tiny_data(small_train, cfg)
    folds = make_folds(X, y, replace(cfg, cv_splits=2))
    shared = {
        "cfg": cfg,
        "n_trials": 1,
        "space": TINY_SPACE,
        "data": (X, y),
        "folds": folds,
        "es_rounds": TINY_ES_ROUNDS,
        "n_jobs": 1,
        "log_mlflow": False,
        "write_outputs": False,
        "verbose": False,
    }

    first = run_study("xgb", "A", storage=f"sqlite:///{(tmp_path / 'first.db').as_posix()}", **shared)
    second = run_study("xgb", "A", storage=f"sqlite:///{(tmp_path / 'second.db').as_posix()}", **shared)

    print(f"первый запуск: ROC-AUC = {first.best_value:.6f}, параметры = {first.best_params}")
    print(f"второй запуск: ROC-AUC = {second.best_value:.6f}, параметры = {second.best_params}")
    assert first.best_value == second.best_value
    assert first.best_params == second.best_params
    assert first.fold_scores == second.fold_scores


def test_smoke_writes_to_separate_file(small_train, cfg, tmp_path) -> None:
    """Smoke пишет best_params_smoke.json и не попадает в best_params.json полного прогона."""
    X, y = _tiny_data(small_train, cfg)
    tables_dir = tmp_path / "tables"
    run_cfg = replace(cfg, tables_dir=tables_dir, figures_dir=tmp_path / "figures")
    shared = {
        "cfg": run_cfg,
        "n_trials": 1,
        "space": TINY_SPACE,
        "data": (X, y),
        "folds": make_folds(X, y, replace(cfg, cv_splits=2)),
        "es_rounds": TINY_ES_ROUNDS,
        "n_jobs": 1,
        "log_mlflow": False,
        "verbose": False,
    }

    run_study("xgb", "A", smoke=True, storage=f"sqlite:///{(tmp_path / 'smoke.db').as_posix()}", **shared)
    smoke_payload = read_json(tables_dir / "best_params_smoke.json")
    full_path = tables_dir / "best_params.json"
    print(
        f"smoke: ключи в best_params_smoke.json = {sorted(smoke_payload)}, "
        f"best_params.json существует = {full_path.exists()}"
    )
    assert sorted(smoke_payload) == ["xgb_A_smoke"]
    assert not full_path.exists()

    run_study("xgb", "A", smoke=False, storage=f"sqlite:///{(tmp_path / 'full.db').as_posix()}", **shared)
    smoke_payload = read_json(tables_dir / "best_params_smoke.json")
    full_payload = read_json(full_path)
    print(
        f"полный прогон: ключи в best_params.json = {sorted(full_payload)}, "
        f"ключи в best_params_smoke.json = {sorted(smoke_payload)}"
    )
    assert sorted(full_payload) == ["xgb_A"]
    assert sorted(smoke_payload) == ["xgb_A_smoke"]


def test_tag_isolates_new_search(small_train, cfg, tmp_path) -> None:
    """Тег даёт суффикс имени study и ключа, поэтому новая сетка не смешивается с прежним поиском."""
    X, y = _tiny_data(small_train, cfg)
    tables_dir = tmp_path / "tables"
    run_cfg = replace(cfg, tables_dir=tables_dir, figures_dir=tmp_path / "figures")
    shared = {
        "cfg": run_cfg,
        "n_trials": 1,
        "space": TINY_SPACE,
        "data": (X, y),
        "folds": make_folds(X, y, replace(cfg, cv_splits=2)),
        "es_rounds": TINY_ES_ROUNDS,
        "n_jobs": 1,
        "log_mlflow": False,
        "verbose": False,
    }

    tagged = run_study("xgb", "A", tag="wide", storage=f"sqlite:///{(tmp_path / 'tagged.db').as_posix()}", **shared)
    tagged_smoke = run_study(
        "xgb",
        "A",
        smoke=True,
        tag="wide",
        storage=f"sqlite:///{(tmp_path / 'tagged_smoke.db').as_posix()}",
        **shared,
    )
    full_payload = read_json(tables_dir / "best_params.json")
    smoke_payload = read_json(tables_dir / "best_params_smoke.json")

    print(f"study = {tagged.study_name}, ключи в best_params.json = {sorted(full_payload)}")
    print(f"study = {tagged_smoke.study_name}, ключи в best_params_smoke.json = {sorted(smoke_payload)}")
    assert tagged.study_name == "xgb_A_wide"
    assert tagged_smoke.study_name == "xgb_A_smoke_wide"
    assert sorted(full_payload) == ["xgb_A_wide"]
    assert sorted(smoke_payload) == ["xgb_A_smoke_wide"]
    assert (tmp_path / "figures" / "optuna_history_xgb_A_wide.png").exists()

    try:
        normalize_tag("плохой тег")
    except SystemExit as error:
        print(f"недопустимый тег отклонён: {error}")
    else:
        raise AssertionError("тег с пробелом и кириллицей должен отклоняться")


def test_parse_args_reads_tag() -> None:
    """CLI: без --tag суффикса нет, явный тег нормализуется (регистр и разделитель)."""
    plain = parse_args([])
    tagged = parse_args(["--model", "xgb", "--variant", "A", "--trials", "100", "--tag", "Wide"])
    print(f"без тега: tag = {plain.tag!r}, суффикс = {normalize_tag(plain.tag)!r}")
    print(f"с тегом: tag = {tagged.tag!r}, суффикс = {normalize_tag(tagged.tag)!r}, trials = {tagged.trials}")

    assert plain.tag == ""
    assert normalize_tag(plain.tag) == ""
    assert normalize_tag(tagged.tag) == "_wide"
    assert tagged.trials == 100
