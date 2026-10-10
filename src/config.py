"""Загрузка конфигурации проекта из configs/*.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass(frozen=True)
class Config:
    seed: int
    n_jobs: int
    tune_n_jobs: int
    raw_data: Path
    processed_dir: Path
    split_indices: Path
    figures_dir: Path
    tables_dir: Path
    logs_dir: Path
    mlflow_db: Path
    mlflow_artifacts: Path
    optuna_db: Path
    target: str
    id_column: str
    test_size: float
    stratify: bool
    cv_splits: int
    cv_repeats: int
    inner_valid_size: float
    n_trials: int
    smoke_trials: int
    logreg_trials: int
    es_rounds: int
    sampler: str
    pruner: str
    metric: str
    mlflow_experiment: str
    models: tuple[str, ...]


@dataclass(frozen=True)
class BusinessConfig:
    retention_offer_cost: float
    customer_lifetime_value: float
    retention_success_rate: float
    sensitivity: dict[str, list[float]]


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_tune_jobs(value: Any) -> int:
    """n_jobs для тюнинга: null в конфиге означает половину логических ядер (минимум 1)."""
    if value is None:
        return max(1, (os.cpu_count() or 1) // 2)
    return max(1, int(value))


def load_config(path: Path | None = None) -> Config:
    cfg_path = path or CONFIG_DIR / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    paths, data, cv, tuning = raw["paths"], raw["data"], raw["cv"], raw["tuning"]
    return Config(
        seed=int(raw["seed"]),
        n_jobs=int(raw.get("n_jobs", 1)),
        tune_n_jobs=_resolve_tune_jobs(tuning.get("n_jobs")),
        raw_data=_resolve(paths["raw_data"]),
        processed_dir=_resolve(paths["processed_dir"]),
        split_indices=_resolve(paths["split_indices"]),
        figures_dir=_resolve(paths["figures_dir"]),
        tables_dir=_resolve(paths["tables_dir"]),
        logs_dir=_resolve(paths["logs_dir"]),
        mlflow_db=_resolve(paths["mlflow_db"]),
        mlflow_artifacts=_resolve(paths["mlflow_artifacts"]),
        optuna_db=_resolve(paths.get("optuna_db", "reports/optuna.db")),
        target=str(data["target"]),
        id_column=str(data["id_column"]),
        test_size=float(data["test_size"]),
        stratify=bool(data["stratify"]),
        cv_splits=int(cv["n_splits"]),
        cv_repeats=int(cv["n_repeats"]),
        inner_valid_size=float(cv["inner_valid_size"]),
        n_trials=int(tuning["n_trials"]),
        smoke_trials=int(tuning["smoke_trials"]),
        logreg_trials=int(tuning.get("logreg_trials", 20)),
        es_rounds=int(tuning.get("es_rounds", 50)),
        sampler=str(tuning["sampler"]),
        pruner=str(tuning["pruner"]),
        metric=str(tuning["metric"]),
        mlflow_experiment=str(tuning.get("mlflow_experiment", "churn-boosting-benchmark")),
        models=tuple(str(m) for m in raw["models"]),
    )


def load_business_config(path: Path | None = None) -> BusinessConfig:
    cfg_path = path or CONFIG_DIR / "business.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    sensitivity = {
        key: [float(v) for v in values]
        for key, values in raw.get("sensitivity", {}).items()
    }
    return BusinessConfig(
        retention_offer_cost=float(raw["retention_offer_cost"]),
        customer_lifetime_value=float(raw["customer_lifetime_value"]),
        retention_success_rate=float(raw["retention_success_rate"]),
        sensitivity=sensitivity,
    )
