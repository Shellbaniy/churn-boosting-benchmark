"""Тюнинг гиперпараметров (этап 5): Optuna на train, тестовая выборка не читается.

Протокол одного trial:
  1. внешние фолды те же, что в baseline: StratifiedKFold(5, shuffle=True, random_state=42);
  2. train-часть фолда стратифицированно делится 80/20 на inner-train и inner-val;
  3. модель обучается на inner-train с early stopping по inner-val;
  4. score trial — ROC-AUC на внешнем валидационном фолде, который в early stopping не участвует.
Значение trial — среднее score по 5 фолдам; best_iteration каждого фолда сохраняется.

Препроцессинг (OHE в варианте A, нативные категории в варианте B) обучается на inner-train
внутри Pipeline, поэтому inner-val подаётся в модель уже преобразованным: отдельный transform
не даёт статистике валидации попасть ни в список категорий, ни в параметры scaling.

Запуск по отдельности: python -m src.tune --model xgb --variant A --trials 100
Все studies по очереди: python -m src.tune (make tune / make tune-smoke).
Тег отделяет новую сетку от прежней: python -m src.tune --tag wide даёт study xgb_A_wide
и ключ xgb_A_wide в best_params.json, старые (без тега) результаты остаются на месте.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")  # фигуры сохраняются в файлы, интерактивный backend не нужен
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from lightgbm import early_stopping as lgb_early_stopping  # noqa: E402
from optuna.pruners import MedianPruner  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402
from optuna.trial import FrozenTrial, TrialState  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from xgboost.callback import EarlyStopping as XGBEarlyStopping  # noqa: E402

from src.config import CONFIG_DIR, Config, load_config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.pipelines import (  # noqa: E402
    MODEL_LABELS,
    MODEL_NAMES,
    VARIANT_A,
    VARIANT_B,
    available_variants,
    build_pipeline,
    make_folds,
)
from src.utils import configure_utf8_output, ensure_dir, read_json, set_global_seed, write_json  # noqa: E402

# Подсказку MLflow о встроенном skill глушим: логи прогонов идут в reports/logs.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

SPACE_FILE = CONFIG_DIR / "search_spaces.yaml"
VARIANT_B_SECTION = "variant_b"
SMOKE_SUFFIX = "_smoke"
TAG_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
# LogisticRegression: чтобы trial не зависел от сходимости солвера, итерации фиксируем.
FIXED_LOGREG_PARAMS: dict[str, Any] = {"max_iter": 5000}
WARN_TOTAL_SEC = 3 * 3600
STUDY_SPECS: tuple[tuple[str, str], ...] = (
    ("xgb", VARIANT_A),
    ("xgb", VARIANT_B),
    ("lgbm", VARIANT_A),
    ("lgbm", VARIANT_B),
    ("catboost", VARIANT_A),
    ("catboost", VARIANT_B),
    ("logreg", VARIANT_A),
)
RUN_STATUS = {TrialState.COMPLETE: "FINISHED", TrialState.PRUNED: "KILLED", TrialState.FAIL: "FAILED"}
TRIAL_STATE_LABELS = {
    TrialState.RUNNING: "идёт",
    TrialState.COMPLETE: "завершён",
    TrialState.PRUNED: "pruned",
    TrialState.FAIL: "ошибка",
}
def normalize_tag(tag: str | None) -> str:
    """Тег -> суффикс имени study: --tag wide даёт xgb_A_wide, пустой тег — пустую строку."""
    value = (tag or "").strip().lower()
    if not value:
        return ""
    if not TAG_PATTERN.match(value):
        raise SystemExit(f"Тег {value!r} содержит недопустимые символы: разрешены буквы, цифры, _ и -.")
    return f"_{value}"


def sqlite_uri(path: Path) -> str:
    """URI SQLite-файла для Optuna и MLflow (posix-путь важен на Windows)."""
    return f"sqlite:///{path.as_posix()}"


def resolve_space(model_name: str, variant: str, path: Path | None = None) -> dict[str, Any]:
    """Пространство поиска модели из configs/search_spaces.yaml.

    Для варианта B к общим параметрам добавляются нативные (секция variant_b).
    """
    space_path = path or SPACE_FILE
    raw = yaml.safe_load(space_path.read_text(encoding="utf-8"))
    if model_name not in raw:
        raise ValueError(f"В {space_path} нет пространства для модели {model_name!r}.")
    space = {str(name): dict(spec) for name, spec in raw[model_name].items()}
    if variant == VARIANT_B:
        extras = raw.get(VARIANT_B_SECTION, {}).get(model_name, {})
        space.update({str(name): dict(spec) for name, spec in extras.items()})
    return space


def sample_params(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
    """Сэмплирование одного набора гиперпараметров по описанию из YAML."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = str(spec["type"])
        step = int(spec.get("step", 1))
        log = bool(spec.get("log", False))
        if kind == "int":
            if log and step != 1:
                raise ValueError(f"{name}: лог-шкала несовместима со step={step}.")
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=step, log=log)
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=log)
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"{name}: неизвестный тип распределения {kind!r}.")
    return params


def early_stopping_constructor_params(model_name: str, es_rounds: int) -> dict[str, Any]:
    """Параметры early stopping, которые xgboost принимает только в конструкторе.

    В xgboost 3.x нет ни early_stopping_rounds, ни callbacks в fit: терпение задаётся
    callback-ом, который следит за AUC на eval_set и оставляет лучшую итерацию (save_best).
    """
    if model_name != "xgb":
        return {}
    return {
        "eval_metric": "auc",
        "callbacks": [XGBEarlyStopping(rounds=es_rounds, metric_name="auc", save_best=True, maximize=True)],
    }


def inner_split(y_train: pd.Series, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Стратифицированное 80/20 внутри train-части фолда: позиции 0..len(y_train)-1."""
    positions = np.arange(len(y_train))
    inner_train, inner_valid = train_test_split(
        positions, test_size=cfg.inner_valid_size, stratify=y_train, random_state=cfg.seed
    )
    return np.asarray(inner_train), np.asarray(inner_valid)


def best_iteration_of(estimator: Any) -> int | None:
    """best_iteration обученной модели (CatBoostNative держит модель внутри себя)."""
    inner = getattr(estimator, "model_", estimator)
    for attr in ("best_iteration", "best_iteration_"):
        value = getattr(inner, attr, None)
        if value is not None:
            return int(value)
    getter = getattr(inner, "get_best_iteration", None)
    if callable(getter):
        return int(getter())
    return None


def fit_estimator(
    estimator: Any,
    model_name: str,
    X_fit: Any,
    y_fit: pd.Series,
    X_es: Any,
    y_es: pd.Series,
    es_rounds: int,
) -> None:
    """Обучение с early stopping по inner-val; logreg обучается без early stopping."""
    if model_name == "xgb":
        estimator.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    elif model_name == "lgbm":
        estimator.fit(
            X_fit,
            y_fit,
            eval_set=[(X_es, y_es)],
            eval_metric="auc",
            callbacks=[lgb_early_stopping(es_rounds, verbose=False)],
        )
    elif model_name == "catboost":
        estimator.fit(
            X_fit, y_fit, eval_set=[(X_es, y_es)], use_best_model=True, early_stopping_rounds=es_rounds
        )
    else:
        estimator.fit(X_fit, y_fit)


def fit_fold(
    model_name: str,
    variant: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    inner_train: np.ndarray,
    inner_valid: np.ndarray,
    cfg: Config,
    es_rounds: int,
) -> dict[str, Any]:
    """Один фолд trial: препроцессинг на inner-train, early stopping на inner-val, score на фолде."""
    pipeline = build_pipeline(
        model_name, variant, {**params, **early_stopping_constructor_params(model_name, es_rounds)}, cfg
    )
    preprocess = pipeline[:-1]
    estimator = pipeline[-1]

    X_outer_train, y_outer_train = X.iloc[train_idx], y.iloc[train_idx]
    X_inner_train, y_inner_train = X_outer_train.iloc[inner_train], y_outer_train.iloc[inner_train]
    X_inner_valid, y_inner_valid = X_outer_train.iloc[inner_valid], y_outer_train.iloc[inner_valid]

    started = time.perf_counter()
    preprocess.fit(X_inner_train, y_inner_train)
    fit_estimator(
        estimator,
        model_name,
        preprocess.transform(X_inner_train),
        y_inner_train,
        preprocess.transform(X_inner_valid),
        y_inner_valid,
        es_rounds,
    )
    fit_sec = time.perf_counter() - started

    y_valid = y.iloc[valid_idx]
    proba = estimator.predict_proba(preprocess.transform(X.iloc[valid_idx]))[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_valid, proba)),
        "best_iteration": best_iteration_of(estimator),
        "n_inner_train": int(len(inner_train)),
        "n_inner_valid": int(len(inner_valid)),
        "n_valid": int(len(valid_idx)),
        "fit_sec": round(fit_sec, 3),
    }


def make_objective(
    model_name: str,
    variant: str,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    cfg: Config,
    space: dict[str, Any],
    es_rounds: int,
) -> Callable[[optuna.Trial], float]:
    """Objective для Optuna: средний ROC-AUC по внешним фолдам с pruning по фолдам."""

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, space)
        if model_name == "logreg":
            params.update(FIXED_LOGREG_PARAMS)
            trial.set_user_attr("fixed_params", FIXED_LOGREG_PARAMS)
        scores: list[float] = []
        iterations: list[int | None] = []
        for fold_id, (train_idx, valid_idx) in enumerate(folds):
            inner_train, inner_valid = inner_split(y.iloc[train_idx], cfg)
            fold = fit_fold(
                model_name,
                variant,
                params,
                X,
                y,
                train_idx,
                valid_idx,
                inner_train,
                inner_valid,
                cfg,
                es_rounds,
            )
            scores.append(fold["roc_auc"])
            iterations.append(fold["best_iteration"])
            trial.set_user_attr("fold_roc_auc", [round(score, 6) for score in scores])
            trial.set_user_attr("fold_best_iteration", iterations)
            trial.report(float(np.mean(scores)), fold_id)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


def format_duration(seconds: float) -> str:
    """Время в удобном виде: секунды, минуты или часы."""
    if seconds < 90:
        return f"{seconds:.1f} с"
    if seconds < 5400:
        return f"{seconds / 60:.1f} мин"
    return f"{seconds / 3600:.2f} ч"


def make_trial_printer(n_trials: int) -> Callable[[optuna.Study, FrozenTrial], None]:
    """Печать по каждому trial: номер, значение, время, признак pruned."""

    def print_trial(study: optuna.Study, trial: FrozenTrial) -> None:
        if trial.state == TrialState.RUNNING:  # callback вызывается и на старте trial
            return
        duration = trial.duration.total_seconds() if trial.duration is not None else 0.0
        value = "—" if trial.value is None else f"{trial.value:.4f}"
        state = TRIAL_STATE_LABELS.get(trial.state, trial.state.name)
        print(f"trial {trial.number + 1}/{n_trials}: ROC-AUC = {value}, {format_duration(duration)}, {state}")

    return print_trial


def trials_frame(study: optuna.Study) -> pd.DataFrame:
    """Таблица trials: значение, score по фолдам, best_iteration, время и параметры."""
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        scores = [float(score) for score in trial.user_attrs.get("fold_roc_auc", [])]
        iterations = list(trial.user_attrs.get("fold_best_iteration", []))
        row: dict[str, Any] = {
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "mean_roc_auc": round(float(np.mean(scores)), 6) if scores else np.nan,
            "std_roc_auc": round(float(np.std(scores, ddof=1)), 6) if len(scores) > 1 else np.nan,
            "n_folds": len(scores),
            "fold_roc_auc": json.dumps([round(score, 6) for score in scores]) if scores else "",
            "fold_best_iteration": json.dumps(iterations) if iterations else "",
            "duration_sec": round(trial.duration.total_seconds(), 2) if trial.duration is not None else np.nan,
        }
        for name, value in trial.params.items():
            row[f"param_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class StudyResult:
    """Итог одного study: лучший trial, его фолды и служебные числа."""

    model: str
    variant: str
    study_name: str
    best_value: float
    best_params: dict[str, Any]
    fold_scores: list[float]
    fold_best_iterations: list[int | None]
    n_trials: int
    n_pruned: int
    duration_sec: float
    mean_trial_sec: float

    def as_summary(self) -> dict[str, Any]:
        iterations = [int(value) for value in self.fold_best_iterations if value is not None]
        return {
            "model": self.model,
            "label": MODEL_LABELS[self.model],
            "variant": self.variant,
            "study": self.study_name,
            "best_roc_auc": round(self.best_value, 6),
            "roc_auc_std_folds": (
                round(float(np.std(self.fold_scores, ddof=1)), 6) if len(self.fold_scores) > 1 else None
            ),
            "fold_roc_auc": [round(score, 6) for score in self.fold_scores],
            "best_params": self.best_params,
            "median_best_iteration": int(np.median(iterations)) if iterations else None,
            "n_trials": self.n_trials,
            "n_pruned": self.n_pruned,
            "duration_sec": round(self.duration_sec, 2),
            "mean_trial_sec": round(self.mean_trial_sec, 3),
        }


def save_figures(
    study: optuna.Study,
    model_name: str,
    variant: str,
    smoke: bool,
    figures_dir: Path,
    tag_suffix: str = "",
) -> list[Path]:
    """История оптимизации и важность параметров в reports/figures/optuna_*.png."""
    suffix = (SMOKE_SUFFIX if smoke else "") + tag_suffix
    paths: list[Path] = []
    history_path = figures_dir / f"optuna_history_{model_name}_{variant}{suffix}.png"
    axes = optuna.visualization.matplotlib.plot_optimization_history(study)
    axes.figure.savefig(history_path, dpi=150, bbox_inches="tight")
    plt.close(axes.figure)
    paths.append(history_path)

    importance_path = figures_dir / f"optuna_importance_{model_name}_{variant}{suffix}.png"
    try:
        axes = optuna.visualization.matplotlib.plot_param_importances(study)
    except Exception as error:  # мало завершённых trials (smoke) или один параметр
        print(f"Важность параметров не построена ({importance_path.name}): {error}")
        return paths
    axes.figure.savefig(importance_path, dpi=150, bbox_inches="tight")
    plt.close(axes.figure)
    paths.append(importance_path)
    return paths


def _as_param(value: Any) -> str:
    """MLflow хранит параметры строками."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class MlflowTrialCallback:
    """Свой MLflow-callback: родительский run на study, дочерние run на trial.

    optuna-integration не используется: параметры и значение trial пишутся вручную через
    MlflowClient, активный run не держится, поэтому логирование не мешает optuna.
    """

    def __init__(self, experiment: str, tracking_uri: str, study_name: str, artifact_location: Path) -> None:
        import mlflow  # локальный импорт: тесты и smoke-проверки не должны писать в reports/mlflow.db

        self.client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        found = self.client.get_experiment_by_name(experiment)
        if found is None:
            self.experiment_id = self.client.create_experiment(experiment, artifact_location=str(artifact_location))
        else:
            self.experiment_id = found.experiment_id
        self.parent_run_id = self.client.create_run(self.experiment_id, run_name=study_name).info.run_id
        self._child_runs: dict[int, str] = {}

    def log_study_params(self, params: dict[str, Any]) -> None:
        for name, value in params.items():
            self.client.log_param(self.parent_run_id, name, _as_param(value))

    def __call__(self, study: optuna.Study, trial: FrozenTrial) -> None:
        if trial.state == TrialState.RUNNING:
            run = self.client.create_run(
                self.experiment_id,
                run_name=f"trial_{trial.number:04d}",
                tags={"mlflow.parentRunId": self.parent_run_id},
            )
            run_id = run.info.run_id
            self._child_runs[trial.number] = run_id
            for name, value in trial.params.items():
                self.client.log_param(run_id, name, _as_param(value))
            return

        run_id = self._child_runs.pop(trial.number, None)
        if run_id is None:
            return
        for index, score in enumerate(trial.user_attrs.get("fold_roc_auc", [])):
            self.client.log_metric(run_id, f"fold_{index}_roc_auc", float(score))
        for index, iteration in enumerate(trial.user_attrs.get("fold_best_iteration", [])):
            if iteration is not None:
                self.client.log_metric(run_id, f"fold_{index}_best_iteration", float(iteration))
        if trial.value is not None:
            self.client.log_metric(run_id, "roc_auc", float(trial.value))
        if trial.duration is not None:
            self.client.log_metric(run_id, "duration_sec", trial.duration.total_seconds())
        self.client.set_tag(run_id, "trial_state", trial.state.name)
        self.client.set_terminated(run_id, status=RUN_STATUS.get(trial.state, "FAILED"))

    def finish(self, study: optuna.Study, duration_sec: float) -> None:
        """Итог study: лучшее значение, число trials, время."""
        pruned = sum(1 for trial in study.trials if trial.state == TrialState.PRUNED)
        self.client.log_metric(self.parent_run_id, "best_roc_auc", float(study.best_value))
        self.client.log_metric(self.parent_run_id, "n_trials", float(len(study.trials)))
        self.client.log_metric(self.parent_run_id, "n_pruned", float(pruned))
        self.client.log_metric(self.parent_run_id, "duration_sec", float(duration_sec))
        for name, value in study.best_trial.params.items():
            self.client.log_param(self.parent_run_id, f"best_{name}", _as_param(value))
        self.client.set_tag(self.parent_run_id, "best_roc_auc", f"{study.best_value:.6f}")
        self.client.set_terminated(self.parent_run_id, status="FINISHED")


def run_study(
    model_name: str,
    variant: str,
    *,
    cfg: Config | None = None,
    n_trials: int | None = None,
    smoke: bool = False,
    tag: str = "",
    space: dict[str, Any] | None = None,
    data: tuple[pd.DataFrame, pd.Series] | None = None,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
    storage: str | None = None,
    es_rounds: int | None = None,
    n_jobs: int | None = None,
    log_mlflow: bool = True,
    write_outputs: bool = True,
    verbose: bool = True,
) -> StudyResult:
    """Один study: TPE + MedianPruner, протокол early stopping внутри фолдов, MLflow, артефакты.

    Параметры data/folds/storage/es_rounds/n_jobs нужны тестам: они позволяют прогнать
    тот же протокол на мини-выборке, не записывая ничего в reports/.
    tag добавляет суффикс к имени study, имени trials.csv, картинкам и ключу в best_params.json,
    поэтому новая сетка не смешивается с прежними результатами.
    """
    configure_utf8_output()
    cfg = cfg or load_config()
    variant = str(variant).upper()
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Неизвестная модель {model_name!r}, допустимо: {list(MODEL_NAMES)}.")
    if variant not in available_variants(model_name):
        raise ValueError(f"У модели {model_name!r} нет варианта {variant!r}.")
    budget = int(
        n_trials
        if n_trials is not None
        else (cfg.smoke_trials if smoke else (cfg.logreg_trials if model_name == "logreg" else cfg.n_trials))
    )
    space = space if space is not None else resolve_space(model_name, variant)
    es_rounds = int(es_rounds if es_rounds is not None else cfg.es_rounds)
    tune_jobs = int(n_jobs if n_jobs is not None else cfg.tune_n_jobs)
    fit_cfg = replace(cfg, n_jobs=tune_jobs)
    tag_suffix = normalize_tag(tag)
    study_name = f"{model_name}_{variant}{SMOKE_SUFFIX if smoke else ''}{tag_suffix}"
    # Smoke-прогон не должен перетирать итоговые best_params полного прогона.
    best_params_file = f"best_params{SMOKE_SUFFIX if smoke else ''}.json"

    if data is None:
        train_frame = load_split(cfg)[0]
        data = (train_frame.drop(columns=[cfg.target]), train_frame[cfg.target])
    X, y = data
    folds = folds if folds is not None else make_folds(X, y, cfg)

    set_global_seed(cfg.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if storage is None:
        ensure_dir(cfg.optuna_db.parent)
        storage_url = sqlite_uri(cfg.optuna_db)
    else:
        storage_url = storage
    tracking_uri = ""
    if log_mlflow:
        ensure_dir(cfg.mlflow_db.parent)
        ensure_dir(cfg.mlflow_artifacts)
        tracking_uri = sqlite_uri(cfg.mlflow_db)

    if verbose:
        train_sizes = sorted({len(train_idx) for train_idx, _ in folds})
        valid_sizes = sorted({len(valid_idx) for _, valid_idx in folds})
        print(f"study {study_name}: {MODEL_LABELS[model_name]}, вариант {variant}, seed = {cfg.seed}, потоки = {tune_jobs}")
        print(f"train: {len(X)} строк, {X.shape[1]} признаков, доля оттока = {float(y.mean()):.3f}")
        print(
            f"фолды: {len(folds)} × (train {train_sizes}, valid {valid_sizes}); "
            f"внутри фолда inner-train/inner-val = {1 - cfg.inner_valid_size:.0%}/{cfg.inner_valid_size:.0%}, "
            f"early stopping patience = {es_rounds}"
        )
        print(f"бюджет: {budget} trials, TPE(seed={cfg.seed}) + MedianPruner, storage = {storage_url}")
        if tag_suffix:
            print(
                f"тег {tag_suffix.lstrip('_')}: ключ {study_name} в {best_params_file}, "
                f"trials и картинки тоже с суффиксом {tag_suffix}; прежние результаты не перетираются."
            )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=cfg.seed),
        pruner=MedianPruner(n_startup_trials=cfg.smoke_trials, n_warmup_steps=0),
    )
    existing = len(study.trials)
    remaining = max(0, budget - existing)
    if verbose:
        prefix = f"в study уже {existing} trials," if existing else "новый study,"
        print(f"{prefix} запускаем {remaining}.")

    callbacks: list[Callable[[optuna.Study, FrozenTrial], None]] = []
    if verbose:
        callbacks.append(make_trial_printer(budget))
    mlflow_callback: MlflowTrialCallback | None = None
    if log_mlflow:
        mlflow_callback = MlflowTrialCallback(
            experiment=cfg.mlflow_experiment,
            tracking_uri=tracking_uri,
            study_name=study_name,
            artifact_location=cfg.mlflow_artifacts,
        )
        mlflow_callback.log_study_params(
            {
                "model": model_name,
                "variant": variant,
                "study_name": study_name,
                "smoke": smoke,
                "tag": tag_suffix.lstrip("_") or "none",
                "seed": cfg.seed,
                "n_trials_budget": budget,
                "n_folds": len(folds),
                "inner_valid_size": cfg.inner_valid_size,
                "es_rounds": es_rounds,
                "n_jobs": tune_jobs,
                "train_rows": len(X),
            }
        )
        callbacks.append(mlflow_callback)

    started = time.perf_counter()
    if remaining:
        study.optimize(
            make_objective(model_name, variant, X, y, folds, fit_cfg, space, es_rounds),
            n_trials=remaining,
            callbacks=callbacks,
        )
    duration_sec = time.perf_counter() - started
    if mlflow_callback is not None:
        mlflow_callback.finish(study, duration_sec)

    best = study.best_trial
    durations = [trial.duration.total_seconds() for trial in study.trials if trial.duration is not None]
    result = StudyResult(
        model=model_name,
        variant=variant,
        study_name=study_name,
        best_value=float(study.best_value),
        best_params=dict(best.params),
        fold_scores=[float(score) for score in best.user_attrs.get("fold_roc_auc", [])],
        fold_best_iterations=list(best.user_attrs.get("fold_best_iteration", [])),
        n_trials=len(study.trials),
        n_pruned=sum(1 for trial in study.trials if trial.state == TrialState.PRUNED),
        duration_sec=duration_sec,
        mean_trial_sec=float(np.mean(durations)) if durations else 0.0,
    )
    summary = result.as_summary()
    if verbose:
        print(
            f"итог {study_name}: лучший ROC-AUC = {result.best_value:.4f} "
            f"(std по фолдам = {summary['roc_auc_std_folds']}), trials = {result.n_trials}, pruned = {result.n_pruned}"
        )
        print(f"лучшие параметры: {json.dumps(result.best_params, ensure_ascii=False, sort_keys=True)}")
        print(f"best_iteration по фолдам: {result.fold_best_iterations}, медиана = {summary['median_best_iteration']}")
        print(
            f"время study = {format_duration(result.duration_sec)}, "
            f"среднее время trial = {format_duration(result.mean_trial_sec)}"
        )
        estimate = result.mean_trial_sec * budget
        print(
            f"оценка полного прогона: {budget} trials × {result.mean_trial_sec:.1f} с = "
            f"{format_duration(estimate)} на study; {len(STUDY_SPECS)} studies ≈ "
            f"{format_duration(estimate * len(STUDY_SPECS))}"
        )
        if estimate > WARN_TOTAL_SEC:
            print(
                f"ПРЕДУПРЕЖДЕНИЕ: оценка времени study больше 3 часов ({format_duration(estimate)}) — "
                "запускайте исследования по отдельности (--model) или снижайте бюджет."
            )

    if write_outputs:
        ensure_dir(cfg.tables_dir)
        ensure_dir(cfg.figures_dir)
        trials_path = cfg.tables_dir / f"optuna_trials_{study_name}.csv"
        trials_frame(study).to_csv(trials_path, index=False)
        best_path = cfg.tables_dir / best_params_file
        payload = read_json(best_path) if best_path.exists() else {}
        payload[study_name] = summary
        write_json(best_path, payload)
        figure_paths = save_figures(study, model_name, variant, smoke, cfg.figures_dir, tag_suffix)
        if verbose:
            print(f"сохранено: {trials_path}")
            for path in [best_path, *figure_paths]:
                print(f"сохранено: {path}")
    return result


def studies_for(model_name: str | None, variant: str | None) -> list[tuple[str, str]]:
    """Список studies: все, одна модель (оба варианта) или один вариант одной модели."""
    if model_name is None:
        if variant is not None:
            raise SystemExit("--variant указывается только вместе с --model.")
        return list(STUDY_SPECS)
    if variant is None:
        return [(model_name, available) for available in available_variants(model_name)]
    return [(model_name, variant)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    configure_utf8_output()  # --help и сообщения об ошибках печатают кириллицу до старта study
    parser = argparse.ArgumentParser(description="Optuna-тюнинг на train (тест не используется).")
    parser.add_argument("--model", choices=list(MODEL_NAMES), help="модель; без флага запускаются все studies")
    parser.add_argument("--variant", choices=[VARIANT_A, VARIANT_B], help="вариант пайплайна; по умолчанию оба")
    parser.add_argument("--trials", type=int, default=None, help="бюджет trials на study (по умолчанию из config.yaml)")
    parser.add_argument("--smoke", action="store_true", help="быстрый прогон: 5 trials и суффикс _smoke в именах studies")
    parser.add_argument(
        "--tag",
        default="",
        help="суффикс имени study и ключа в best_params.json (например --tag wide); по умолчанию пусто",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[StudyResult]:
    """Последовательный прогон studies: 6 бустингов (A/B) и небольшой поиск для logreg."""
    args = parse_args(argv)
    cfg = load_config()
    if args.smoke:
        print(f"Режим smoke: {cfg.smoke_trials} trials на study, имена studies с суффиксом {SMOKE_SUFFIX}.")
    tag_suffix = normalize_tag(args.tag)
    if tag_suffix:
        print(
            f"Тег {tag_suffix.lstrip('_')}: studies, trials.csv, картинки и ключи в best_params.json "
            f"получат суффикс {tag_suffix}; прежние результаты (без тега) не перетираются."
        )
    train_frame = load_split(cfg)[0]
    data = (train_frame.drop(columns=[cfg.target]), train_frame[cfg.target])
    print(
        f"Тюнинг только на train: {len(data[0])} строк, {data[0].shape[1]} признаков, "
        f"доля оттока = {float(data[1].mean()):.3f}; тестовая выборка не используется."
    )
    results: list[StudyResult] = []
    for model_name, variant in studies_for(args.model, args.variant):
        results.append(
            run_study(
                model_name,
                variant,
                cfg=cfg,
                n_trials=args.trials,
                smoke=args.smoke,
                tag=args.tag,
                data=data,
            )
        )
    if len(results) > 1:
        print("\nСводка studies, лучший ROC-AUC (среднее по фолдам):")
        for result in results:
            print(
                f"  {result.study_name:<24} {result.best_value:.4f}  "
                f"(trials = {result.n_trials}, pruned = {result.n_pruned}, {format_duration(result.duration_sec)})"
            )
    return results


if __name__ == "__main__":
    main()
