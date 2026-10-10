"""Этап 6: финальный бенчмарк на отложенном тесте — единственное чтение теста.

Протокол:
1. out-of-fold предсказания на train (StratifiedKFold, те же фолды, что в baseline);
2. порог каждой модели подбирается только по train OOF (максимум эмпирической прибыли);
3. обучение на всём train и предсказание теста — здесь тест читается один раз;
4. метрики качества и бизнес-метрики на тесте, эталонные стратегии прибыли, чувствительность;
5. фигуры и краткий вывод о величине различий относительно шума (значимость — этап 7).

Финальными считаются широкие конфигурации (ключи ``*_wide`` в ``best_params.json``), для
LogisticRegression — ``logreg_A``. У tuned бустингов число деревьев фиксируется значением
``median_best_iteration``; early stopping при финальном обучении и при OOF не используется.
Прибыль считается по фактическим меткам: ``profit_i = y_i * CLV * success_rate - offer_cost``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import BusinessConfig, Config, load_business_config, load_config
from src.data import load_split
from src.pipelines import (
    DUMMY_MODEL,
    MODEL_LABELS,
    available_variants,
    build_pipeline,
    make_folds,
)
from src.utils import configure_utf8_output, ensure_dir, read_json, set_global_seed, write_json

plt.rcParams["axes.unicode_minus"] = False

BEST_PARAMS_FILE = "best_params.json"
FINAL_TAG = "wide"
LOGREG_BEST_KEY = "logreg_A"
BOOSTING_MODELS: tuple[str, ...] = ("xgb", "lgbm", "catboost")
TREE_PARAM = {"xgb": "n_estimators", "lgbm": "n_estimators", "catboost": "iterations"}
TREE_SOURCE = "median_best_iteration"
# Дублируется намеренно: бенчмарк не должен тянуть optuna и mlflow ради одной константы,
# совпадение с src.tune.FIXED_LOGREG_PARAMS проверяет тест.
TUNED_LOGREG_FIXED: dict[str, Any] = {"max_iter": 5000}
DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.01 * step, 2) for step in range(1, 100))
TOP_FRACTION = 0.10


@dataclass(frozen=True)
class ModelSpec:
    """Одна строка бенчмарка: модель, вариант пайплайна и источник параметров."""

    key: str
    model_name: str
    variant: str
    kind: str  # dummy | default | tuned | reference
    params: dict[str, Any] | None
    best_key: str | None
    label: str

    @property
    def uses_threshold(self) -> bool:
        """DummyClassifier без порога: для него считаются только ROC-AUC, PR-AUC и Brier."""
        return self.model_name != DUMMY_MODEL


def profit_at_threshold(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    threshold: float,
    clv: float,
    offer_cost: float,
    success_rate: float,
) -> float:
    """Эмпирическая прибыль при пороге: контактируем клиентов с ``p_i >= threshold``.

    ``profit_i = y_i * CLV * success_rate - offer_cost``. Используются фактические метки ``y_i``,
    а не вероятности ``p_i``: калибровка бустингов не проверялась, поэтому ``p_i`` как веса
    дали бы смещённую прибыль (уточнение к PLAN.md, этап 6, п. 2).
    """
    y = np.asarray(y_true, dtype=float)
    contacted = np.asarray(proba, dtype=float) >= threshold
    retained = float(y[contacted].sum())
    return retained * clv * success_rate - float(contacted.sum()) * offer_cost


def profit_curve(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    clv: float,
    offer_cost: float,
    success_rate: float,
    grid: tuple[float, ...] = THRESHOLD_GRID,
) -> np.ndarray:
    """Прибыль для каждого порога сетки (одна точка на порог, без интерполяции)."""
    return np.array(
        [profit_at_threshold(y_true, proba, threshold, clv, offer_cost, success_rate) for threshold in grid],
        dtype=float,
    )


def select_threshold(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    clv: float,
    offer_cost: float,
    success_rate: float,
    grid: tuple[float, ...] = THRESHOLD_GRID,
) -> tuple[float, float]:
    """Порог с максимальной прибылью по OOF-предсказаниям train.

    На вход идут только out-of-fold метки и вероятности train, поэтому тест в подбор порога
    попасть не может. При равной прибыли выбирается меньший порог (первый максимум сетки):
    это больше контактов и полностью детерминированный выбор.
    """
    profits = profit_curve(y_true, proba, clv, offer_cost, success_rate, grid)
    best_index = int(np.argmax(profits))
    return float(grid[best_index]), float(profits[best_index])


def final_best_keys() -> list[str]:
    """Финальные конфигурации: широкие поиски бустингов (``*_wide``) и ``logreg_A``."""
    keys = [
        f"{model}_{variant}_{FINAL_TAG}"
        for model in BOOSTING_MODELS
        for variant in available_variants(model)
    ]
    keys.append(LOGREG_BEST_KEY)
    return keys


def read_best_params(path: Path) -> dict[str, Any]:
    """Читает ``best_params.json`` (этап 5) и объясняет, что делать при отсутствии файла."""
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла финальных гиперпараметров: {path}. Сначала выполните `make tune-one` "
            f"с --tag {FINAL_TAG} для всех бустингов."
        )
    return read_json(path)


def tuned_params(model_name: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Параметры tuned-модели: ``best_params`` и фиксированное число деревьев.

    Число деревьев берётся из ``median_best_iteration`` (медиана best_iteration по фолдам
    тюнинга), поэтому финальное обучение идёт без early stopping и без eval_set. Значения
    ``n_estimators``/``iterations``, сэмплированные в широком поиске, перекрываются медианой.
    """
    params = dict(entry["best_params"])
    if model_name == "logreg":
        params.update(TUNED_LOGREG_FIXED)
        return params
    tree_param = TREE_PARAM[model_name]
    median = entry.get(TREE_SOURCE)
    if median is None:
        raise ValueError(
            f"{model_name}: в best_params.json нет {TREE_SOURCE}, число деревьев задать нельзя."
        )
    params[tree_param] = int(median)
    return params


def build_specs(payload: dict[str, Any]) -> list[ModelSpec]:
    """15 моделей бенчмарка: dummy, logreg default/tuned и 6 бустингов default/tuned."""
    expected = final_best_keys()
    missing = [key for key in expected if key not in payload]
    if missing:
        raise KeyError(
            f"В best_params.json нет финальных конфигураций: {missing}. Ожидаются ключи "
            f"{expected}; для бустингов это результат широкого поиска (--tag {FINAL_TAG})."
        )

    specs = [
        ModelSpec(DUMMY_MODEL, DUMMY_MODEL, "A", "default", None, None, MODEL_LABELS[DUMMY_MODEL]),
        ModelSpec(
            "logreg_default", "logreg", "A", "default", None, None, f"{MODEL_LABELS['logreg']} default"
        ),
        ModelSpec(
            "logreg_tuned",
            "logreg",
            "A",
            "tuned",
            tuned_params("logreg", payload[LOGREG_BEST_KEY]),
            LOGREG_BEST_KEY,
            f"{MODEL_LABELS['logreg']} tuned ({LOGREG_BEST_KEY})",
        ),
    ]
    for model in BOOSTING_MODELS:
        for variant in available_variants(model):
            base = f"{model}_{variant}"
            best_key = f"{base}_{FINAL_TAG}"
            specs.append(
                ModelSpec(
                    f"{base}_default",
                    model,
                    variant,
                    "default",
                    None,
                    None,
                    f"{MODEL_LABELS[model]} {variant} default",
                )
            )
            specs.append(
                ModelSpec(
                    f"{base}_tuned",
                    model,
                    variant,
                    "tuned",
                    tuned_params(model, payload[best_key]),
                    best_key,
                    f"{MODEL_LABELS[model]} {variant} tuned ({best_key})",
                )
            )
    return specs


def make_pipeline_for(spec: ModelSpec, cfg: Config) -> Pipeline:
    """Pipeline модели: препроцессинг внутри Pipeline, seed и n_jobs как в baseline (1)."""
    return build_pipeline(spec.model_name, spec.variant, spec.params, cfg)


def positive_class_column(estimator: Any) -> int:
    """Индекс столбца вероятности положительного класса в ``predict_proba``.

    Если положительного класса (оттока) не было в обучающей части, вероятность для него
    не определена: молча подставлять нули нельзя, поэтому это ошибка, а не warning.
    """
    classes = list(getattr(estimator, "classes_", []))
    if 1 not in classes:
        raise ValueError(
            f"в обучающей части фолда нет положительного класса: classes_ = {classes}, "
            f"вероятность оттока посчитать нельзя"
        )
    return classes.index(1)


def oof_predictions(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    cfg: Config,
) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold вероятности на train: препроцессор и модель переобучаются на каждой train-части."""
    proba = np.empty(len(X), dtype=float)
    fit_seconds: list[float] = []
    for train_idx, valid_idx in folds:
        pipeline = make_pipeline_for(spec, cfg)
        started = time.perf_counter()
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        fit_seconds.append(time.perf_counter() - started)
        column = positive_class_column(pipeline[-1])
        proba[valid_idx] = pipeline.predict_proba(X.iloc[valid_idx])[:, column]
    return proba, fit_seconds


def fit_on_train(spec: ModelSpec, X: pd.DataFrame, y: pd.Series, cfg: Config) -> tuple[Pipeline, float]:
    """Обучение модели на всём train один раз (без early stopping); возвращает пайплайн и время."""
    pipeline = make_pipeline_for(spec, cfg)
    started = time.perf_counter()
    pipeline.fit(X, y)
    return pipeline, time.perf_counter() - started


def predict_proba_test(pipeline: Pipeline, X_test: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Вероятности положительного класса на тесте и время инференса."""
    column = positive_class_column(pipeline[-1])
    started = time.perf_counter()
    proba = pipeline.predict_proba(X_test)[:, column]
    return np.asarray(proba, dtype=float), time.perf_counter() - started


def ranking_metrics(y_true: np.ndarray | pd.Series, proba: np.ndarray) -> tuple[float, float]:
    """lift@10% и recall@10%: top-10% по рангу вероятности относительно базовой доли оттока."""
    y = np.asarray(y_true, dtype=float)
    scores = np.asarray(proba, dtype=float)
    k = max(1, int(round(len(scores) * TOP_FRACTION)))
    order = np.argsort(-scores, kind="stable")[:k]
    hits = float(y[order].sum())
    base_rate = float(y.mean())
    lift = (hits / k) / base_rate
    recall = hits / float(y.sum())
    return float(lift), float(recall)


def binary_metrics(
    y_true: np.ndarray | pd.Series, proba: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """F1/recall/precision и число контактов при фиксированном пороге."""
    pred = (np.asarray(proba, dtype=float) >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "contacts": int(pred.sum()),
        "contact_rate": float(pred.mean()),
    }


_TEST_READS = 0


def load_test_frame(cfg: Config) -> pd.DataFrame:
    """Единственное чтение теста за запуск: строки ``test_idx`` из ``data/processed/split_indices.npz``.

    Счётчик страхует протокол: если тест понадобился второй раз, это ошибка, а не повод
    перечитать файл (все последующие этапы работают с сохранённым test_predictions.parquet).
    """
    global _TEST_READS
    _TEST_READS += 1
    if _TEST_READS > 1:
        raise RuntimeError(
            f"Тест читается повторно (обращение №{_TEST_READS}): используйте сохранённый "
            f"test_predictions.parquet, тест читается ровно один раз за запуск."
        )
    return load_split(cfg)[1]


def reset_test_reads() -> None:
    """Сбрасывает счётчик чтений теста (нужно тестам; в бенчмарке не используется)."""
    global _TEST_READS
    _TEST_READS = 0


def metrics_row(
    spec: ModelSpec,
    y_test: pd.Series,
    proba_test: np.ndarray,
    threshold: float | None,
    n_train: int,
    business: BusinessConfig,
    fit_sec: float,
    infer_sec: float,
    oof_fit_seconds: list[float],
) -> dict[str, Any]:
    """Строка таблицы метрик на тесте: качество, бизнес-метрики при двух порогах и время."""
    y = np.asarray(y_test, dtype=float)
    row: dict[str, Any] = {
        "key": spec.key,
        "label": spec.label,
        "model": spec.model_name,
        "variant": spec.variant,
        "kind": spec.kind,
        "n_train": int(n_train),
        "n_test": int(len(y)),
        "test_churn_rate": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, proba_test)), 4),
        "pr_auc": round(float(average_precision_score(y, proba_test)), 4),
        "brier": round(float(brier_score_loss(y, proba_test)), 4),
        "log_loss": round(float(log_loss(y, proba_test, labels=[0, 1])), 4),
        "threshold": None,
        "threshold_source": None,
        "profit_at_threshold": None,
        "profit_per_customer_at_threshold": None,
        "contacts_at_threshold": None,
        "contact_rate_at_threshold": None,
        "f1_at_threshold": None,
        "recall_at_threshold": None,
        "precision_at_threshold": None,
        "f1_at_0.5": None,
        "recall_at_0.5": None,
        "precision_at_0.5": None,
        "contacts_at_0.5": None,
        "lift_at_10pct": None,
        "recall_at_10pct": None,
        "oof_fit_sec_mean": round(float(np.mean(oof_fit_seconds)), 4) if oof_fit_seconds else None,
        "fit_sec_full_train": round(fit_sec, 2),
        "infer_sec_test": round(infer_sec, 4),
    }
    if not spec.uses_threshold or threshold is None:
        return row

    clv = business.customer_lifetime_value
    cost = business.retention_offer_cost
    rate = business.retention_success_rate
    profit = profit_at_threshold(y, proba_test, threshold, clv, cost, rate)
    at_threshold = binary_metrics(y, proba_test, threshold)
    at_half = binary_metrics(y, proba_test, DEFAULT_THRESHOLD)
    lift, recall_top = ranking_metrics(y, proba_test)

    row.update(
        {
            "threshold": threshold,
            "threshold_source": "oof_train",
            "profit_at_threshold": round(profit, 2),
            "profit_per_customer_at_threshold": round(profit / len(y), 4),
            "contacts_at_threshold": at_threshold["contacts"],
            "contact_rate_at_threshold": round(float(at_threshold["contact_rate"]), 4),
            "f1_at_threshold": round(float(at_threshold["f1"]), 4),
            "recall_at_threshold": round(float(at_threshold["recall"]), 4),
            "precision_at_threshold": round(float(at_threshold["precision"]), 4),
            "f1_at_0.5": round(float(at_half["f1"]), 4),
            "recall_at_0.5": round(float(at_half["recall"]), 4),
            "precision_at_0.5": round(float(at_half["precision"]), 4),
            "contacts_at_0.5": at_half["contacts"],
            "lift_at_10pct": round(lift, 4),
            "recall_at_10pct": round(recall_top, 4),
        }
    )
    return row


def reference_rows(y_test: pd.Series, business: BusinessConfig) -> list[dict[str, Any]]:
    """Эталонные стратегии прибыли: не контактировать никого (0) и контактировать всех."""
    y = np.asarray(y_test, dtype=float)
    clv = business.customer_lifetime_value
    cost = business.retention_offer_cost
    rate = business.retention_success_rate
    churners = int(y.sum())
    all_profit = churners * clv * rate - len(y) * cost

    base: dict[str, Any] = {
        "model": "reference",
        "variant": "",
        "kind": "reference",
        "n_train": None,
        "n_test": int(len(y)),
        "test_churn_rate": round(float(y.mean()), 4),
        "threshold": None,
        "threshold_source": "reference",
    }
    return [
        {
            **base,
            "key": "reference_contact_none",
            "label": "Не контактировать никого",
            "profit_at_threshold": 0.0,
            "profit_per_customer_at_threshold": 0.0,
            "contacts_at_threshold": 0,
            "contact_rate_at_threshold": 0.0,
        },
        {
            **base,
            "key": "reference_contact_all",
            "label": "Контактировать всех",
            "profit_at_threshold": round(all_profit, 2),
            "profit_per_customer_at_threshold": round(all_profit / len(y), 4),
            "contacts_at_threshold": int(len(y)),
            "contact_rate_at_threshold": 1.0,
        },
    ]


def sensitivity_rows(
    spec: ModelSpec,
    y_oof: np.ndarray,
    proba_oof: np.ndarray,
    y_test: pd.Series,
    proba_test: np.ndarray,
    business: BusinessConfig,
    rates: list[float],
) -> list[dict[str, Any]]:
    """Чувствительность прибыли: для каждого success_rate порог заново подбирается по OOF."""
    y_test_array = np.asarray(y_test, dtype=float)
    clv = business.customer_lifetime_value
    cost = business.retention_offer_cost
    rows: list[dict[str, Any]] = []
    for rate in rates:
        threshold, oof_profit = select_threshold(y_oof, proba_oof, clv, cost, rate)
        test_profit = profit_at_threshold(y_test_array, proba_test, threshold, clv, cost, rate)
        rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "success_rate": round(float(rate), 4),
                "threshold_oof": threshold,
                "oof_profit": round(oof_profit, 2),
                "test_profit": round(test_profit, 2),
                "test_profit_per_customer": round(test_profit / len(y_test_array), 4),
                "n_contacts": int((np.asarray(proba_test) >= threshold).sum()),
            }
        )
    return rows


def reference_sensitivity_rows(
    y_test: pd.Series, business: BusinessConfig, rates: list[float]
) -> list[dict[str, Any]]:
    """Те же эталонные стратегии, но по сетке success_rate: порог здесь не нужен."""
    y = np.asarray(y_test, dtype=float)
    clv = business.customer_lifetime_value
    cost = business.retention_offer_cost
    churners = int(y.sum())
    rows: list[dict[str, Any]] = []
    for rate in rates:
        all_profit = churners * clv * rate - len(y) * cost
        for key, label, profit, contacts in (
            ("reference_contact_none", "Не контактировать никого", 0.0, 0),
            ("reference_contact_all", "Контактировать всех", all_profit, len(y)),
        ):
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "success_rate": round(float(rate), 4),
                    "threshold_oof": None,
                    "oof_profit": None,
                    "test_profit": round(profit, 2),
                    "test_profit_per_customer": round(profit / len(y), 4),
                    "n_contacts": contacts,
                }
            )
    return rows


def benchmark_table(
    specs: list[ModelSpec],
    y_test: pd.Series,
    test_proba: dict[str, np.ndarray],
    thresholds: dict[str, dict[str, Any]],
    oof_seconds: dict[str, list[float]],
    fit_seconds: dict[str, float],
    infer_seconds: dict[str, float],
    n_train: int,
    business: BusinessConfig,
) -> pd.DataFrame:
    """Таблица метрик на тесте: модели плюс эталонные стратегии, колонки строго TABLE_COLUMNS."""
    rows: list[dict[str, Any]] = [
        metrics_row(
            spec,
            y_test,
            test_proba[spec.key],
            thresholds.get(spec.key, {}).get("threshold"),
            n_train,
            business,
            fit_seconds[spec.key],
            infer_seconds[spec.key],
            oof_seconds[spec.key],
        )
        for spec in specs
    ]
    rows.extend(reference_rows(y_test, business))
    return pd.DataFrame([{column: row.get(column) for column in TABLE_COLUMNS} for row in rows])


def predictions_frame(y_test: pd.Series, test_proba: dict[str, np.ndarray]) -> pd.DataFrame:
    """Кадр test_predictions.parquet: фактические метки теста и вероятности всех моделей.

    Имя без префикса ``test_``: после импорта в тест-модуль pytest иначе собирал бы её как тест.
    """
    frame = pd.DataFrame({"y_true": y_test.to_numpy()})
    for key, proba in test_proba.items():
        frame[f"p_{key}"] = proba
    return frame


def build_sensitivity_table(
    specs: list[ModelSpec],
    y_oof: np.ndarray,
    oof_proba: dict[str, np.ndarray],
    y_test: pd.Series,
    test_proba: dict[str, np.ndarray],
    business: BusinessConfig,
    rates: list[float],
) -> pd.DataFrame:
    """Чувствительность прибыли: для каждого success_rate порог заново подбирается по OOF train."""
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.uses_threshold:
            continue
        rows.extend(
            sensitivity_rows(spec, y_oof, oof_proba[spec.key], y_test, test_proba[spec.key], business, rates)
        )
    rows.extend(reference_sensitivity_rows(y_test, business, rates))
    return pd.DataFrame(rows)


def save_test_artifacts(
    table: pd.DataFrame,
    test_frame: pd.DataFrame,
    sensitivity_table: pd.DataFrame,
    tables_dir: Path,
) -> tuple[Path, Path, Path]:
    """Сохраняет test_benchmark.csv, test_predictions.parquet и business_sensitivity.csv."""
    ensure_dir(tables_dir)
    benchmark_path = tables_dir / "test_benchmark.csv"
    table.to_csv(benchmark_path, index=False)
    test_path = tables_dir / "test_predictions.parquet"
    test_frame.to_parquet(test_path, index=False)
    sensitivity_path = tables_dir / "business_sensitivity.csv"
    sensitivity_table.to_csv(sensitivity_path, index=False)
    return benchmark_path, test_path, sensitivity_path


def make_figures(
    specs: list[ModelSpec],
    oof_proba: dict[str, np.ndarray],
    test_proba: dict[str, np.ndarray],
    y_oof: np.ndarray,
    y_test: np.ndarray,
    business: BusinessConfig,
    figures_dir: Path,
) -> list[Path]:
    """Три фигуры этапа: калибровка на тесте, прибыль от порога (OOF и тест), PR-кривые."""
    ensure_dir(figures_dir)
    return [
        plot_calibration(specs, test_proba, y_test, figures_dir / "test_calibration.png"),
        plot_profit_curves(
            specs, oof_proba, test_proba, y_oof, y_test, business, figures_dir / "test_profit_curves.png"
        ),
        plot_pr_curves(specs, test_proba, y_test, figures_dir / "test_pr_curves.png"),
    ]


def display_specs(specs: list[ModelSpec]) -> list[ModelSpec]:
    """Модели для фигур: tuned-конфигурации плюс logreg default как ориентир.

    Все 14 пороговых моделей на одной фигуре нечитаемы, поэтому кривые показываются для
    tuned-моделей (шесть бустингов и logreg_tuned), а logreg_default добавлен для сравнения.
    Числа по всем 15 моделям всё равно лежат в test_benchmark.csv.
    """
    chosen = [spec for spec in specs if spec.kind == "tuned"]
    chosen.extend(spec for spec in specs if spec.key == "logreg_default")
    return chosen


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_calibration(
    specs: list[ModelSpec], test_proba: dict[str, np.ndarray], y_test: np.ndarray, path: Path
) -> Path:
    """Reliability curves на тесте: предсказанная вероятность против фактической доли оттока."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="идеальная калибровка")
    for spec in specs:
        fraction, mean_predicted = calibration_curve(
            y_test, test_proba[spec.key], n_bins=10, strategy="uniform"
        )
        ax.plot(mean_predicted, fraction, marker="o", markersize=3, label=spec.label)
    ax.set_xlabel("Средняя предсказанная вероятность")
    ax.set_ylabel("Фактическая доля оттока в бине")
    ax.set_title("Calibration curves на тесте")
    ax.legend(fontsize=7, loc="upper left")
    return _save_figure(fig, path)


def plot_profit_curves(
    specs: list[ModelSpec],
    oof_proba: dict[str, np.ndarray],
    test_proba: dict[str, np.ndarray],
    y_oof: np.ndarray,
    y_test: np.ndarray,
    business: BusinessConfig,
    path: Path,
) -> Path:
    """Прибыль на клиента от порога: слева OOF train (где порог выбирается), справа тест."""
    clv = business.customer_lifetime_value
    cost = business.retention_offer_cost
    rate = business.retention_success_rate
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panels = (
        (axes[0], oof_proba, y_oof, "OOF train (выбор порога)"),
        (axes[1], test_proba, y_test, "тест (оценка)"),
    )
    for ax, proba, y, title in panels:
        for spec in specs:
            curve = profit_curve(y, proba[spec.key], clv, cost, rate)
            ax.plot(THRESHOLD_GRID, curve / len(y), label=spec.label)
        ax.axhline(0.0, color="grey", linestyle=":")
        ax.set_xlabel("Порог контакта")
        ax.set_ylabel("Прибыль на клиента")
        ax.set_title(f"Прибыль от порога: {title}")
    axes[1].legend(fontsize=7, loc="lower right")
    return _save_figure(fig, path)


def plot_pr_curves(
    specs: list[ModelSpec], test_proba: dict[str, np.ndarray], y_test: np.ndarray, path: Path
) -> Path:
    """Precision-Recall кривые на тесте с PR-AUC в легенде."""
    fig, ax = plt.subplots(figsize=(7, 6))
    base_rate = float(np.mean(y_test))
    ax.axhline(base_rate, color="grey", linestyle="--", label=f"базовая доля оттока = {base_rate:.3f}")
    for spec in specs:
        precision, recall, _ = precision_recall_curve(y_test, test_proba[spec.key])
        pr_auc = average_precision_score(y_test, test_proba[spec.key])
        ax.plot(recall, precision, label=f"{spec.label}: PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall на тесте")
    ax.legend(fontsize=7, loc="lower left")
    return _save_figure(fig, path)


DEFAULT_SENSITIVITY: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
INT_COLUMNS = frozenset(
    {"n_train", "n_test", "contacts_at_threshold", "contacts_at_0.5", "n_contacts"}
)
# Текстовые колонки печатаются как есть: без этого format_table пытается привести их к float.
TEXT_COLUMNS = frozenset({"key", "label", "model", "variant", "kind", "threshold_source"})
# Колонки test_benchmark.csv и вид таблиц для печати. Список задан явно, чтобы одна опечатка
# в имени метрики не «теряла» колонку молча.
TABLE_COLUMNS: tuple[str, ...] = (
    "key",
    "label",
    "model",
    "variant",
    "kind",
    "n_train",
    "n_test",
    "test_churn_rate",
    "roc_auc",
    "pr_auc",
    "brier",
    "log_loss",
    "threshold",
    "threshold_source",
    "profit_at_threshold",
    "profit_per_customer_at_threshold",
    "contacts_at_threshold",
    "contact_rate_at_threshold",
    "f1_at_threshold",
    "recall_at_threshold",
    "precision_at_threshold",
    "f1_at_0.5",
    "recall_at_0.5",
    "precision_at_0.5",
    "contacts_at_0.5",
    "lift_at_10pct",
    "recall_at_10pct",
    "oof_fit_sec_mean",
    "fit_sec_full_train",
    "infer_sec_test",
)
QUALITY_VIEW: tuple[str, ...] = (
    "key",
    "label",
    "kind",
    "roc_auc",
    "pr_auc",
    "brier",
    "log_loss",
    "threshold",
    "f1_at_threshold",
    "recall_at_threshold",
    "precision_at_threshold",
    "f1_at_0.5",
    "recall_at_0.5",
    "precision_at_0.5",
)
BUSINESS_VIEW: tuple[str, ...] = (
    "key",
    "label",
    "kind",
    "threshold",
    "profit_at_threshold",
    "profit_per_customer_at_threshold",
    "contacts_at_threshold",
    "contact_rate_at_threshold",
    "recall_at_threshold",
    "precision_at_threshold",
    "lift_at_10pct",
    "recall_at_10pct",
)


def format_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Человекочитаемый вид: фиксированная точность по типу колонки, пропуски как «-»."""
    def fmt_one(value: Any, column: str) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        if column in TEXT_COLUMNS:
            return str(value)
        if column in INT_COLUMNS:
            return f"{int(value)}"
        if column == "profit_at_threshold":
            return f"{float(value):,.0f}"
        if column in ("oof_fit_sec_mean", "fit_sec_full_train"):
            return f"{float(value):.1f}"
        if column == "threshold" or column == "threshold_oof":
            return f"{float(value):.2f}"
        return f"{float(value):.4f}"

    return pd.DataFrame(
        {column: [fmt_one(value, column) for value in frame[column]] for column in columns},
        index=frame.index,
    )


def print_benchmark_report(
    table: pd.DataFrame,
    sensitivity_table: pd.DataFrame,
    payload: dict[str, Any],
    specs: list[ModelSpec],
) -> None:
    """Печатает две таблицы (качество и бизнес), разброс относительно шума и чувствительность."""
    models = table[table["kind"] != "reference"].copy()
    ordered_quality = models.sort_values(
        ["roc_auc", "profit_at_threshold"], ascending=[False, False]
    )
    ordered_all = table.sort_values(
        ["roc_auc", "profit_at_threshold"], ascending=[False, False], na_position="last"
    )

    print()
    print("Таблица 1. Качество на тесте (сортировка по ROC-AUC, затем по прибыли):")
    print(format_table(ordered_quality, QUALITY_VIEW).to_string(index=False))

    print()
    print("Таблица 2. Бизнес-метрики на тесте (порог из OOF train; эталонные стратегии в конце):")
    print(format_table(ordered_all, BUSINESS_VIEW).to_string(index=False))

    best_roc = ordered_quality.iloc[0]
    best_profit = models.sort_values("profit_at_threshold", ascending=False).iloc[0]
    fold_stds = [
        payload[spec.best_key]["roc_auc_std_folds"]
        for spec in specs
        if spec.best_key and payload.get(spec.best_key, {}).get("roc_auc_std_folds")
    ]
    mean_fold_std = float(np.mean(fold_stds)) if fold_stds else float("nan")
    spread = float(models["roc_auc"].max() - models["roc_auc"].min())
    n_test = int(models["n_test"].iloc[0])
    n_churn = int(round(float(models["test_churn_rate"].iloc[0]) * n_test))
    verdict = "в пределах фолдового шума" if spread <= mean_fold_std else "шире фолдового шума"

    print()
    print("Вывод по величине различий:")
    print(
        f"  лучший по ROC-AUC: {best_roc['key']} = {best_roc['roc_auc']:.4f}; "
        f"лучший по прибыли: {best_profit['key']} = {best_profit['profit_at_threshold']:,.0f}"
    )
    print(
        f"  разброс ROC-AUC между 14 пороговыми моделями на тесте = {spread:.4f}, "
        f"средний разброс ROC-AUC по фолдам тюнинга = {mean_fold_std:.4f} -> {verdict}"
    )
    print(
        f"  тест: {n_test} строк, клиентов с оттоком = {n_churn}: разница ROC-AUC в 0.001 "
        f"достигается перестановкой единиц наблюдений, поэтому гонка по 4-му знаку смысла не имеет"
    )

    top_key = str(ordered_quality.iloc[0]["key"])
    context = sensitivity_table[sensitivity_table["key"].isin([top_key, "reference_contact_all"])]
    print()
    print(f"Чувствительность прибыли к success_rate: {top_key} против «контактировать всех»:")
    print(
        format_table(
            context.sort_values(["key", "success_rate"]),
            ("key", "success_rate", "threshold_oof", "test_profit", "test_profit_per_customer", "n_contacts"),
        ).to_string(index=False)
    )


def run_benchmark(cfg: Config | None = None) -> pd.DataFrame:
    """Полный протокол этапа 6: OOF → пороги по OOF → один прогон теста → таблицы и фигуры."""
    configure_utf8_output()
    cfg = cfg or load_config()
    business = load_business_config()
    set_global_seed(cfg.seed)
    reset_test_reads()

    payload = read_best_params(cfg.tables_dir / BEST_PARAMS_FILE)
    specs = build_specs(payload)
    tuned = display_specs(specs)
    rates = [float(rate) for rate in business.sensitivity.get("retention_success_rate", DEFAULT_SENSITIVITY)]

    train = load_split(cfg)[0]
    X = train.drop(columns=[cfg.target])
    y = train[cfg.target]
    y_oof = y.to_numpy()
    folds = make_folds(X, y, cfg)

    print("Этап 6: финальный бенчмарк на отложенном тесте")
    print(f"Train: строк = {len(X)}, признаков = {X.shape[1]}, доля оттока = {y.mean():.4f}")
    print(f"Моделей = {len(specs)}: dummy, logreg default/tuned и 6 бустинговых конфигураций default/tuned")
    print(
        f"OOF-фолды: StratifiedKFold(n_splits={cfg.cv_splits}, shuffle=True, random_state={cfg.seed}) — "
        f"те же, что в baseline"
    )
    print(
        f"Финальные конфигурации: поиски с тегом '{FINAL_TAG}' у бустингов и {LOGREG_BEST_KEY} у logreg; "
        f"у tuned число деревьев = {TREE_SOURCE}, early stopping выключен"
    )
    print(
        f"Бизнес: CLV = {business.customer_lifetime_value}, offer_cost = {business.retention_offer_cost}, "
        f"success_rate = {business.retention_success_rate}"
    )
    print(f"Сетка порогов: {THRESHOLD_GRID[0]:.2f}–{THRESHOLD_GRID[-1]:.2f} с шагом 0.01")

    print()
    print("Шаг 1. Out-of-fold предсказания на train:")
    oof_proba: dict[str, np.ndarray] = {}
    oof_seconds: dict[str, list[float]] = {}
    oof_frame = pd.DataFrame({"y_true": y_oof})
    for spec in specs:
        proba, seconds = oof_predictions(spec, X, y, folds, cfg)
        oof_proba[spec.key] = proba
        oof_seconds[spec.key] = seconds
        oof_frame[f"p_{spec.key}"] = proba
        print(
            f"  {spec.key:<18} ROC-AUC = {roc_auc_score(y_oof, proba):.4f}, "
            f"обучение = {np.mean(seconds):.2f} с/фолд"
        )

    print()
    print("Шаг 2. Пороги из OOF train (максимум прибыли на фактических метках):")
    thresholds: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if not spec.uses_threshold:
            print(f"  {spec.key:<18} без порога (dummy: только ROC-AUC, PR-AUC, Brier)")
            continue
        threshold, oof_profit = select_threshold(
            y_oof,
            oof_proba[spec.key],
            business.customer_lifetime_value,
            business.retention_offer_cost,
            business.retention_success_rate,
        )
        contacts = int((oof_proba[spec.key] >= threshold).sum())
        thresholds[spec.key] = {
            "label": spec.label,
            "model": spec.model_name,
            "variant": spec.variant,
            "kind": spec.kind,
            "best_key": spec.best_key,
            "threshold": threshold,
            "oof_profit": round(oof_profit, 2),
            "oof_profit_per_customer": round(oof_profit / len(y_oof), 4),
            "n_contacted": contacts,
            "contact_rate": round(contacts / len(y_oof), 4),
        }
        print(
            f"  {spec.key:<18} порог = {threshold:.2f}, контактов = {contacts} "
            f"({contacts / len(y_oof):.1%}), прибыль OOF = {oof_profit:,.0f}"
        )

    ensure_dir(cfg.tables_dir)
    oof_path = cfg.tables_dir / "oof_predictions.parquet"
    oof_frame.to_parquet(oof_path, index=False)
    thresholds_path = cfg.tables_dir / "thresholds.json"
    write_json(
        thresholds_path,
        {
            "source": (
                f"out-of-fold предсказания train: StratifiedKFold(n_splits={cfg.cv_splits}, "
                f"shuffle=True, random_state={cfg.seed}); тест не использовался"
            ),
            "rule": "максимум прибыли profit_i = y_i * CLV * success_rate - offer_cost по фактическим меткам",
            "grid": {"low": THRESHOLD_GRID[0], "high": THRESHOLD_GRID[-1], "step": 0.01},
            "business": {
                "customer_lifetime_value": business.customer_lifetime_value,
                "retention_offer_cost": business.retention_offer_cost,
                "retention_success_rate": business.retention_success_rate,
            },
            "thresholds": thresholds,
        },
    )
    print(f"  OOF-вероятности: {oof_path}")
    print(f"  Пороги: {thresholds_path}")

    test = load_test_frame(cfg)
    X_test = test.drop(columns=[cfg.target])
    y_test = test[cfg.target]
    print()
    print(
        f"Шаг 3. Тест: строк = {len(X_test)}, признаков = {X_test.shape[1]}, "
        f"доля оттока = {y_test.mean():.4f}, клиентов с оттоком = {int(y_test.sum())}"
    )

    test_proba: dict[str, np.ndarray] = {}
    fit_seconds: dict[str, float] = {}
    infer_seconds: dict[str, float] = {}
    for spec in specs:
        pipeline, fit_sec = fit_on_train(spec, X, y, cfg)
        proba, infer_sec = predict_proba_test(pipeline, X_test)
        test_proba[spec.key] = proba
        fit_seconds[spec.key] = fit_sec
        infer_seconds[spec.key] = infer_sec
        print(
            f"  {spec.key:<18} ROC-AUC = {roc_auc_score(y_test, proba):.4f}, обучение на train = {fit_sec:.1f} с, "
            f"инференс = {infer_sec:.4f} с"
        )

    test_frame = predictions_frame(y_test, test_proba)
    table = benchmark_table(
        specs,
        y_test,
        test_proba,
        thresholds,
        oof_seconds,
        fit_seconds,
        infer_seconds,
        len(X),
        business,
    )

    print()
    print("Шаг 4. Чувствительность к success_rate (порог каждый раз пересчитывается по OOF):")
    sensitivity_table = build_sensitivity_table(specs, y_oof, oof_proba, y_test, test_proba, business, rates)
    benchmark_path, test_path, sensitivity_path = save_test_artifacts(
        table, test_frame, sensitivity_table, cfg.tables_dir
    )
    print(f"  success_rate = {rates}")
    print(f"  строк в таблице чувствительности = {len(sensitivity_table)} -> {sensitivity_path}")

    figures = make_figures(
        tuned, oof_proba, test_proba, y_oof, y_test.to_numpy(), business, cfg.figures_dir
    )

    print_benchmark_report(table, sensitivity_table, payload, specs)

    print()
    print("Сохранено:")
    for path in (benchmark_path, test_path, sensitivity_path, *figures):
        print(f"  {path}")
    return table


def main() -> None:
    """Точка входа `python -m src.evaluate` (цель `make benchmark`)."""
    run_benchmark()


if __name__ == "__main__":
    main()
