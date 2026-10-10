"""Пайплайны A (унифицированный) и B (нативный) и фабрика моделей.

Pipeline A: ColumnTransformer с OHE (handle_unknown="ignore") для категориальных и
passthrough для числовых; StandardScaler только для LogisticRegression.
Pipeline B: категориальные колонки приводятся к нативному представлению (category dtype
для LightGBM/XGBoost, строки с sentinel для CatBoost), числовые не меняются.

Все обучаемые преобразования (список категорий, среднее и std) живут внутри Pipeline и
переобучаются на каждом фолде, поэтому статистика валидации в них попасть не может.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import Config, load_config
from src.data import load_split
from src.utils import configure_utf8_output, set_global_seed

VARIANT_A = "A"
VARIANT_B = "B"
MODEL_NAMES: tuple[str, ...] = ("logreg", "xgb", "lgbm", "catboost")
# Опорная модель baseline: она не входит в MODEL_NAMES, поэтому не участвует в smoke и тюнинге.
DUMMY_MODEL = "dummy"
SUPPORTED_MODELS: tuple[str, ...] = MODEL_NAMES + (DUMMY_MODEL,)
UNKNOWN_TOKEN = "<unknown>"

# Дефолтные параметры моделей менять нельзя (никакого тюнинга), поэтому в фабрике
# фиксируются только seed, число потоков и режим логирования.
MODEL_LABELS = {
    DUMMY_MODEL: "DummyClassifier",
    "logreg": "LogisticRegression",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "catboost": "CatBoost",
}


def numeric_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]


def categorical_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Приводит категориальные колонки к типу, который принимают нативные модели.

    mode="category": dtype category (LightGBM, XGBoost), неизвестное значение -> NaN.
    mode="string": dtype str, неизвестное значение -> unknown_token
    (CatBoost отклоняет NaN в cat_features: "NaN values should be converted to string").

    Список категорий запоминается в fit, поэтому transform никогда не расширяет его
    значениями из валидации: всё неизвестное превращается в NaN/sentinel.
    """

    def __init__(self, mode: str = "category", unknown_token: str = UNKNOWN_TOKEN) -> None:
        self.mode = mode
        self.unknown_token = unknown_token

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CategoricalEncoder":
        if self.mode not in {"category", "string"}:
            raise ValueError(f"Неизвестный mode={self.mode!r}, ожидается 'category' или 'string'.")
        self.columns_ = categorical_columns(X)
        self.categories_ = {
            column: pd.Index(sorted(pd.unique(X[column].dropna()).tolist()))
            for column in self.columns_
        }
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for column, categories in self.categories_.items():
            codes = categories.get_indexer(X[column].astype(object))
            if self.mode == "category":
                X[column] = pd.Categorical.from_codes(codes, categories=categories)
            else:
                known = np.asarray(categories, dtype=object)[np.where(codes < 0, 0, codes)]
                values = np.where(codes < 0, self.unknown_token, known)
                X[column] = pd.Series(values, index=X.index).astype("str")
        return X


class CatBoostNative(BaseEstimator, ClassifierMixin):
    """CatBoost со списком cat_features, который определяется в fit.

    pandas 3 отдаёт строковые колонки как dtype str, а CatBoost их не угадывает:
    без явного cat_features падает с "Cannot convert 'Two year' to float".
    """

    def __init__(
        self,
        random_state: int = 42,
        thread_count: int = 1,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.random_state = random_state
        self.thread_count = thread_count
        self.params = params

    def fit(self, X: pd.DataFrame, y: Any) -> "CatBoostNative":
        self.cat_features_ = categorical_columns(X)
        self.model_ = CatBoostClassifier(
            random_state=self.random_state,
            thread_count=self.thread_count,
            verbose=0,
            allow_writing_files=False,
            cat_features=self.cat_features_,
            **(self.params or {}),
        )
        self.model_.fit(X, y)
        self.classes_ = np.asarray(self.model_.classes_).ravel()
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(X)).ravel()


def build_preprocessor(model_name: str) -> ColumnTransformer:
    """Общий препроцессинг варианта A: числовые passthrough, OHE; scaling только для logreg."""
    numeric_step: Any = StandardScaler() if model_name == "logreg" else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric_step, numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ]
    )


def build_estimator(
    model_name: str,
    variant: str,
    cfg: Config,
    params: dict[str, Any] | None = None,
) -> Any:
    overrides = dict(params or {})
    if model_name == DUMMY_MODEL:
        return DummyClassifier(strategy="prior", random_state=cfg.seed)
    if model_name == "logreg":
        return LogisticRegression(random_state=cfg.seed, **overrides)
    if model_name == "xgb":
        return XGBClassifier(
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
            enable_categorical=variant == VARIANT_B,
            **overrides,
        )
    if model_name == "lgbm":
        return LGBMClassifier(random_state=cfg.seed, n_jobs=cfg.n_jobs, verbose=-1, **overrides)
    if model_name == "catboost":
        if variant == VARIANT_B:
            return CatBoostNative(random_state=cfg.seed, thread_count=cfg.n_jobs, params=overrides)
        return CatBoostClassifier(
            random_state=cfg.seed,
            thread_count=cfg.n_jobs,
            verbose=0,
            allow_writing_files=False,
            **overrides,
        )
    raise ValueError(f"Неизвестная модель {model_name!r}, ожидается одна из {SUPPORTED_MODELS}.")


def available_variants(model_name: str) -> tuple[str, ...]:
    """Нативное представление категорий есть у бустингов, поэтому logreg и dummy только A."""
    if model_name in {"logreg", DUMMY_MODEL}:
        return (VARIANT_A,)
    return (VARIANT_A, VARIANT_B)


def build_pipeline(
    model_name: str,
    variant: str,
    params: dict[str, Any] | None = None,
    cfg: Config | None = None,
) -> Pipeline:
    """Собирает Pipeline для модели и варианта препроцессинга.

    params переопределяет параметры финальной модели (нужно только тестам и smoke-прогонам;
    по умолчанию параметры дефолтные). seed и число потоков берутся из configs/config.yaml.
    """
    cfg = cfg or load_config()
    variant = str(variant).upper()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Неизвестная модель {model_name!r}, ожидается одна из {SUPPORTED_MODELS}.")
    if variant not in {VARIANT_A, VARIANT_B}:
        raise ValueError(f"Неизвестный вариант {variant!r}, ожидается 'A' или 'B'.")
    if variant == VARIANT_B and model_name in {"logreg", DUMMY_MODEL}:
        raise ValueError(
            f"У модели {model_name!r} нет нативного варианта B: используйте вариант A."
        )

    estimator = build_estimator(model_name, variant, cfg, params)
    if variant == VARIANT_A:
        steps = [("preprocess", build_preprocessor(model_name)), ("model", estimator)]
    else:
        mode = "string" if model_name == "catboost" else "category"
        steps = [("categorical", CategoricalEncoder(mode=mode)), ("model", estimator)]
    return Pipeline(steps)


def make_folds(X: pd.DataFrame, y: pd.Series, cfg: Config) -> list[tuple[np.ndarray, np.ndarray]]:
    """Одни и те же фолды для всех моделей и вариантов."""
    cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.seed)
    return list(cv.split(X, y))


def cross_val_scores(
    model_name: str,
    variant: str,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Config,
    folds: list[tuple[np.ndarray, np.ndarray]],
    params: dict[str, Any] | None = None,
) -> tuple[list[float], list[float]]:
    """ROC-AUC по фолдам: препроцессор переобучается на каждом фолде-train."""
    scores: list[float] = []
    times: list[float] = []
    for train_idx, valid_idx in folds:
        pipe = build_pipeline(model_name, variant, params, cfg)
        started = time.perf_counter()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        times.append(time.perf_counter() - started)
        proba = pipe.predict_proba(X.iloc[valid_idx])[:, 1]
        scores.append(float(roc_auc_score(y.iloc[valid_idx], proba)))
    return scores, times


def run_smoke(cfg: Config | None = None) -> pd.DataFrame:
    """Дефолтные модели на одних фолдах, только train, без тюнинга."""
    configure_utf8_output()
    cfg = cfg or load_config()
    set_global_seed(cfg.seed)

    train = load_split(cfg)[0]
    X = train.drop(columns=[cfg.target])
    y = train[cfg.target]
    numeric = numeric_columns(X)
    categorical = categorical_columns(X)

    print("Smoke: обучаем дефолтные модели только на train, тестовая выборка не читается.")
    print(f"Train: строк = {len(X)}, признаков = {X.shape[1]}, доля оттока = {y.mean():.4f}")
    print(f"Числовые признаки ({len(numeric)}): {', '.join(numeric)}")
    print(f"Категориальные признаки ({len(categorical)}): {', '.join(categorical)}")
    folds = make_folds(X, y, cfg)
    print(
        f"Фолды: StratifiedKFold(n_splits={cfg.cv_splits}, shuffle=True, random_state={cfg.seed}) — "
        f"одни и те же для всех моделей и вариантов."
    )
    print(
        f"seed = {cfg.seed}, n_jobs/thread_count = {cfg.n_jobs}, "
        "параметры моделей дефолтные (тюнинга нет), вариант A = OHE + passthrough "
        "(StandardScaler только для logreg), вариант B = нативные категории."
    )

    rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        for variant in available_variants(model_name):
            scores, times = cross_val_scores(model_name, variant, X, y, cfg, folds)
            mean_auc = float(np.mean(scores))
            std_auc = float(np.std(scores, ddof=1))
            print(
                f"{MODEL_LABELS[model_name]:<18} вариант {variant}: ROC-AUC = {mean_auc:.4f} ± "
                f"{std_auc:.4f} (mean ± std, {len(scores)} фолдов), обучение = {np.mean(times):.2f} с/фолд"
            )
            rows.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "roc_auc_mean": round(mean_auc, 4),
                    "roc_auc_std": round(std_auc, 4),
                    "fit_sec_mean": round(float(np.mean(times)), 2),
                }
            )

    result = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    print("\nSmoke сводка (train-only CV, дефолтные параметры):")
    print(result.to_string(index=False))
    print(
        f"\nДиапазон ROC-AUC по дефолтам: {result['roc_auc_mean'].min():.4f}–"
        f"{result['roc_auc_mean'].max():.4f}. Ожидание для Telco — около 0.84–0.85; "
        "значение заметно выше означает, что нужно искать утечку."
    )
    return result


def main() -> None:
    run_smoke()


if __name__ == "__main__":
    main()


