"""Этап 8: SHAP-интерпретация уже выбранных моделей.

Модели, гиперпараметры и пороги здесь не выбираются и не меняются: берутся финальные
конфигурации этапа 6 (ключи ``*_wide`` и ``logreg_A`` из ``best_params.json``, число деревьев =
``median_best_iteration``). Каждая модель обучается на всём train один раз, SHAP считается на
тестовой выборке: это описательная интерпретация — тест используется только для объяснения
предсказаний и на выбор моделей, порогов и гиперпараметров не влияет.

Перед расчётом SHAP предсказания переобученных моделей сверяются с сохранёнными
``reports/tables/test_predictions.parquet`` (этап 6): при расхождении работа останавливается.

SHAP-значения считаются в пространстве лог-шансов (margin/log-odds). Вариант A использует OHE,
поэтому SHAP-значения всех столбцов одного исходного признака складываются (аддитивность
сохраняется) и дальше работа идёт с 19 исходными признаками.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # фигуры сохраняются в файлы, интерактивный backend не нужен

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from catboost import Pool  # noqa: E402
from scipy import stats as sps  # noqa: E402

from src.config import Config, load_config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.evaluate import (  # noqa: E402
    BEST_PARAMS_FILE,
    ModelSpec,
    build_specs,
    fit_on_train,
    load_test_frame,
    predict_proba_test,
    read_best_params,
    reset_test_reads,
)
from src.pipelines import categorical_columns  # noqa: E402
from src.utils import configure_utf8_output, ensure_dir, read_json, set_global_seed  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

MODEL_KEYS: tuple[str, ...] = (
    "xgb_A_tuned",
    "lgbm_A_tuned",
    "catboost_A_tuned",
    "catboost_B_tuned",
    "logreg_tuned",
)
# Основная модель для dependence и waterfall: простая аддитивная модель, выбор сделан заранее.
MAIN_MODEL_KEY = "xgb_A_tuned"
DEPENDENCE_MODELS: tuple[str, ...] = ("xgb_A_tuned", "catboost_B_tuned")
DEPENDENCE_NUMERIC: tuple[str, ...] = ("tenure", "MonthlyCharges")
DEPENDENCE_CATEGORICAL = "Contract"
INTERACTION_MODELS: tuple[str, ...] = ("xgb_A_tuned", "lgbm_A_tuned")
INTERACTION_ROWS = 300
TOP_N = 5
EXPECTED_FEATURES = 19
ATOL_DEFAULT = 1e-6
ATOL_CATBOOST = 1e-4
THRESHOLDS_FILE = "thresholds.json"
PREDICTIONS_FILE = "test_predictions.parquet"
IMPORTANCE_FILE = "shap_importance.csv"
TOP_FEATURES_FILE = "shap_top_features.csv"
SPEARMAN_FILE = "shap_spearman.csv"
OVERLAP_FILE = "shap_overlap_top5.csv"


@dataclass(frozen=True)
class ShapResult:
    """SHAP-значения одной модели: до и после агрегации OHE-столбцов в исходные признаки."""

    key: str
    label: str
    features: list[str]
    values: np.ndarray
    base_value: float
    raw_features: list[str]
    raw_values: np.ndarray


def selected_specs(payload: dict[str, Any]) -> list[ModelSpec]:
    """Пять финальных конфигураций этапа 6; параметры берутся из best_params.json без правок."""
    specs = {spec.key: spec for spec in build_specs(payload)}
    missing = [key for key in MODEL_KEYS if key not in specs]
    if missing:
        raise KeyError(
            f"В best_params.json нет финальных конфигураций: {missing}. Ожидаются ключи "
            f"{list(MODEL_KEYS)} (бустинги — результат широкого поиска, logreg — logreg_A)."
        )
    return [specs[key] for key in MODEL_KEYS]


def transform_matrix(spec: ModelSpec, pipeline: Any, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Матрица признаков после препроцессора и имена её столбцов (вариант B: исходные колонки)."""
    if spec.variant == "B":
        encoder = pipeline.named_steps["categorical"]
        encoded = encoder.transform(frame)
        return np.asarray(encoded), list(encoded.columns)
    preprocessor = pipeline.named_steps["preprocess"]
    matrix = preprocessor.transform(frame)
    return np.asarray(matrix, dtype=float), list(preprocessor.get_feature_names_out())


def source_feature(column: str, categorical: list[str], all_features: Sequence[str]) -> str:
    """Исходный признак для столбца матрицы: ``num__tenure`` → ``tenure``, ``cat__Contract_One year`` → ``Contract``."""
    name = column.split("__", 1)[1] if "__" in column else column
    if name in all_features:
        return name
    matches = [feature for feature in categorical if name.startswith(f"{feature}_")]
    if not matches:
        raise ValueError(f"Столбец {column!r} не относится ни к одному исходному признаку.")
    # Самый длинный совпадающий префикс: имена категориальных признаков не перепутаются.
    return max(matches, key=len)


def aggregate_values(
    values: np.ndarray,
    columns: Sequence[str],
    categorical: list[str],
    features: Sequence[str],
) -> np.ndarray:
    """Складывает SHAP-значения OHE-столбцов одного исходного признака (сумма по строке сохраняется)."""
    index = {feature: position for position, feature in enumerate(features)}
    aggregated = np.zeros((values.shape[0], len(features)), dtype=float)
    for position, column in enumerate(columns):
        aggregated[:, index[source_feature(column, categorical, features)]] += values[:, position]
    return aggregated


def as_2d(values: Any, n_columns: int) -> np.ndarray:
    """Приводит вывод SHAP к матрице (n объектов, n признаков).

    Разные версии SHAP для бинарной задачи отдают то (n, p), то (n, p, 1), то список массивов
    по классам: берётся положительный класс.
    """
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values, dtype=float)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[1] != n_columns:
        raise ValueError(f"SHAP вернул массив формы {array.shape}, ожидалась (n, {n_columns}).")
    return array


def base_value_of(explainer: Any) -> float:
    """Базовое значение (expected value) в лог-шансах: для бинарной задачи берётся последнее."""
    raw = np.ravel(np.asarray(explainer.expected_value, dtype=float))
    if raw.size == 0:
        raise ValueError("у объяснителя нет expected_value")
    return float(raw[-1])


def catboost_model(pipeline: Any) -> Any:
    """Внутренний CatBoostClassifier: у варианта B модель завёрнута в CatBoostNative."""
    inner = pipeline.named_steps["model"]
    return inner.model_ if hasattr(inner, "model_") else inner


def explain_model(spec: ModelSpec, pipeline: Any, frame: pd.DataFrame, features: Sequence[str]) -> ShapResult:
    """SHAP-значения модели на ``frame`` (в лог-шансах) с агрегацией OHE-столбцов.

    xgb/lgbm: ``shap.TreeExplainer`` на матрице после препроцессора;
    catboost: ``model.get_feature_importance(Pool(...), type="ShapValues")``, последний столбец
    результата — базовое значение, он отбрасывается; logreg: ``shap.LinearExplainer`` на матрице.
    """
    categorical = categorical_columns(frame)
    if spec.model_name in {"xgb", "lgbm"}:
        matrix, columns = transform_matrix(spec, pipeline, frame)
        explainer = shap.TreeExplainer(pipeline.named_steps["model"])
        values = as_2d(explainer.shap_values(matrix), len(columns))
        base_value = base_value_of(explainer)
    elif spec.model_name == "logreg":
        matrix, columns = transform_matrix(spec, pipeline, frame)
        explainer = shap.LinearExplainer(pipeline.named_steps["model"], matrix)
        values = as_2d(explainer.shap_values(matrix), len(columns))
        base_value = base_value_of(explainer)
    elif spec.model_name == "catboost":
        model = catboost_model(pipeline)
        if spec.variant == "B":
            encoded = pipeline.named_steps["categorical"].transform(frame)
            columns = list(encoded.columns)
            pool = Pool(encoded, cat_features=categorical_columns(encoded))
        else:
            matrix, columns = transform_matrix(spec, pipeline, frame)
            pool = Pool(matrix, feature_names=columns)
        shap_matrix = np.asarray(model.get_feature_importance(pool, type="ShapValues"), dtype=float)
        if shap_matrix.shape[1] == len(columns) + 1:
            values = shap_matrix[:, :-1]
            base_value = float(shap_matrix[0, -1])
        else:
            values = shap_matrix
            margin = np.asarray(model.predict(pool, prediction_type="RawFormulaVal"), dtype=float).ravel()
            base_value = float(np.mean(margin - values.sum(axis=1)))
    else:
        raise ValueError(f"SHAP не настроен для модели {spec.model_name!r}.")

    aggregated = aggregate_values(values, columns, categorical, features)
    return ShapResult(
        key=spec.key,
        label=spec.label,
        features=list(features),
        values=aggregated,
        base_value=float(base_value),
        raw_features=list(columns),
        raw_values=values,
    )


def as_interaction_3d(values: Any, n_columns: int) -> np.ndarray:
    """Приводит shap_interaction_values к (n объектов, p, p) для бинарной задачи."""
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"ожидался трёхмерный массив взаимодействий, получена форма {array.shape}")
    if array.shape[2] == n_columns:
        array = array[-1] if array.shape[0] == 2 else array
    elif array.shape[1] == n_columns and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.shape[1] != n_columns or array.shape[2] != n_columns:
        raise ValueError(f"форма взаимодействий {array.shape} не соответствует {n_columns} признакам")
    return array


def interaction_diagnostic(
    spec: ModelSpec,
    pipeline: Any,
    frame: pd.DataFrame,
    n_rows: int = INTERACTION_ROWS,
    seed: int = 42,
) -> dict[str, Any]:
    """Доля недиагональных вкладов в ``shap_interaction_values``: проверка аддитивности модели.

    У лучших конфигураций глубина ограничена (max_depth = 1 у xgb, num_leaves = 2 у lgbm),
    поэтому взаимодействий ожидается почти ноль; если API недоступен, диагностика пропускается.
    """
    if spec.model_name not in {"xgb", "lgbm"} or spec.variant != "A":
        return {"key": spec.key, "supported": False, "reason": "только xgb/lgbm варианта A"}
    rng = np.random.default_rng(seed)
    size = min(n_rows, frame.shape[0])
    index = np.sort(rng.choice(frame.shape[0], size=size, replace=False))
    sample = frame.iloc[index]
    matrix, columns = transform_matrix(spec, pipeline, sample)
    try:
        explainer = shap.TreeExplainer(pipeline.named_steps["model"])
        interactions = as_interaction_3d(explainer.shap_interaction_values(matrix), len(columns))
    except Exception as error:  # noqa: BLE001 - API SHAP и бэкендов меняется между версиями
        print(f"  {spec.key:<18} shap_interaction_values недоступен ({error}); диагностика пропущена")
        return {"key": spec.key, "supported": False, "reason": str(error)}
    diagonal = float(np.abs(np.diagonal(interactions, axis1=1, axis2=2)).sum())
    total = float(np.abs(interactions).sum())
    share = (total - diagonal) / total if total > 0 else 0.0
    return {
        "key": spec.key,
        "supported": True,
        "rows": int(interactions.shape[0]),
        "n_matrix_columns": int(interactions.shape[1]),
        "non_diagonal_share": float(share),
    }


def importance_frame(results: Sequence[ShapResult], features: Sequence[str]) -> pd.DataFrame:
    """Средний |SHAP| и знак среднего SHAP по каждому исходному признаку для каждой модели."""
    rows: list[dict[str, Any]] = []
    for result in results:
        if list(result.features) != list(features):
            raise ValueError(f"{result.key}: порядок признаков не совпадает с колонками train.")
        mean_abs = np.abs(result.values).mean(axis=0)
        mean_signed = result.values.mean(axis=0)
        order = np.argsort(-mean_abs, kind="stable")
        ranks = np.empty(mean_abs.size, dtype=int)
        for rank, position in enumerate(order, start=1):
            ranks[position] = rank
        for position, feature in enumerate(features):
            rows.append(
                {
                    "model": result.key,
                    "feature": feature,
                    "mean_abs_shap": float(mean_abs[position]),
                    "mean_shap": float(mean_signed[position]),
                    "rank": int(ranks[position]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("таблица важности пуста: нет ни одного SHAP-результата")
    return frame.round({"mean_abs_shap": 6, "mean_shap": 6})


def top_features_frame(importance: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Топ-``top_n`` признаков по каждой модели."""
    subset = importance.loc[importance["rank"] <= top_n, ["model", "rank", "feature", "mean_abs_shap"]]
    return subset.sort_values(["model", "rank"], kind="stable").reset_index(drop=True)


def rank_correlation(left: Sequence[float], right: Sequence[float]) -> tuple[float, float]:
    """Ранговая корреляция Спирмена и её p-значение: одинаковые ранги → 1, обратные → -1."""
    rho, p_value = sps.spearmanr(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    return float(rho), float(p_value)


def spearman_frame(importance: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Спирмен между важностями моделей по всем исходным признакам: по строке на пару моделей."""
    order = list(dict.fromkeys(importance["feature"].tolist()))
    wide = importance.pivot(index="feature", columns="model", values="mean_abs_shap").loc[order]
    missing = [key for key in keys if key not in wide.columns]
    if missing:
        raise KeyError(f"в таблице важности нет моделей: {missing}")
    rows: list[dict[str, Any]] = []
    for model_a, model_b in combinations(keys, 2):
        rho, p_value = rank_correlation(wide[model_a].to_numpy(), wide[model_b].to_numpy())
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "spearman": round(rho, 4),
                "p_value": round(p_value, 6),
                "n_features": int(wide.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def spearman_matrix(spearman: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Симметричная матрица ρ по всем парам моделей (диагональ 1) для печати и фигуры."""
    matrix = pd.DataFrame(np.eye(len(keys)), index=list(keys), columns=list(keys))
    for row in spearman.itertuples():
        matrix.loc[row.model_a, row.model_b] = float(row.spearman)
        matrix.loc[row.model_b, row.model_a] = float(row.spearman)
    return matrix


def overlap_frame(top: pd.DataFrame, keys: Sequence[str], top_n: int = TOP_N) -> pd.DataFrame:
    """Пересечение топ-``top_n`` между парами моделей."""
    sets = {key: set(top.loc[top["model"] == key, "feature"]) for key in keys}
    rows: list[dict[str, Any]] = []
    for model_a, model_b in combinations(keys, 2):
        common = sorted(sets[model_a] & sets[model_b])
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "n_overlap": len(common),
                "overlap_share": round(len(common) / top_n, 4),
                "common_features": "; ".join(common),
            }
        )
    return pd.DataFrame(rows)


def thresholds_from_json(cfg: Config, specs: Sequence[ModelSpec]) -> dict[str, float]:
    """Пороги моделей из ``thresholds.json`` этапа 6 (подобраны на OOF train, здесь не меняются)."""
    path = cfg.tables_dir / THRESHOLDS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Нет файла порогов: {path}. Сначала выполните `make benchmark`.")
    payload = read_json(path)
    entries = payload.get("thresholds", payload)
    thresholds: dict[str, float] = {}
    for spec in specs:
        if spec.key not in entries:
            raise KeyError(f"в {path} нет порога для {spec.key!r}; доступные записи: {sorted(entries)}")
        entry = entries[spec.key]
        thresholds[spec.key] = float(entry["threshold"] if isinstance(entry, dict) else entry)
    return thresholds


def check_predictions(
    cfg: Config, specs: Sequence[ModelSpec], test_proba: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Сверяет предсказания переобученных моделей с ``test_predictions.parquet`` этапа 6.

    Совпадение означает, что окружение и гиперпараметры те же, что в бенчмарке, поэтому SHAP
    объясняет именно те модели, чьи метрики опубликованы. При расхождении — останов без графиков.
    """
    path = cfg.tables_dir / PREDICTIONS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла предсказаний: {path}. Сначала выполните `make benchmark` (этап 6)."
        )
    stored = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        column = f"p_{spec.key}"
        if column not in stored.columns:
            raise KeyError(f"в {path} нет колонки {column!r}; доступные колонки: {list(stored.columns)}")
        atol = ATOL_CATBOOST if spec.model_name == "catboost" else ATOL_DEFAULT
        diff = float(np.max(np.abs(stored[column].to_numpy(dtype=float) - np.asarray(test_proba[spec.key]))))
        rows.append({"key": spec.key, "max_abs_diff": diff, "atol": atol, "ok": bool(diff <= atol)})
    frame = pd.DataFrame(rows)
    failed = frame.loc[~frame["ok"], "key"].tolist()
    if failed:
        raise RuntimeError(
            f"Предсказания переобученных моделей не совпали с {path}: {failed}. "
            f"SHAP не считается: проверьте, что best_params.json и окружение соответствуют этапу 6."
        )
    return frame


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bar(result: ShapResult, path: Path) -> Path:
    """Средний |SHAP| по 19 исходным признакам, отсортированный по убыванию."""
    mean_abs = np.abs(result.values).mean(axis=0)
    order = np.argsort(mean_abs)
    features = np.asarray(result.features, dtype=object)[order]
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.barh(features, mean_abs[order], color="#1f4e79")
    ax.set_xlabel("Средний |SHAP| (лог-шансы)")
    ax.set_title(f"Важность признаков: {result.label}")
    return _save_figure(fig, path)


def plot_beeswarm(result: ShapResult, path: Path) -> Path:
    """Beeswarm по агрегированным 19 признакам (цвет — сам SHAP, значения в лог-шансах)."""
    shap.summary_plot(
        result.values,
        feature_names=result.features,
        max_display=EXPECTED_FEATURES,
        show=False,
    )
    figure = plt.gcf()
    figure.set_size_inches(8.0, 7.0)
    plt.xlabel("SHAP (лог-шансы)")
    plt.title(f"Распределение SHAP по признакам: {result.label}")
    return _save_figure(figure, path)


def plot_dependence_numeric(result: ShapResult, frame: pd.DataFrame, feature: str, path: Path) -> Path:
    """Scatter зависимости SHAP от значения числового признака плюс средний SHAP по децилям."""
    index = result.features.index(feature)
    values = frame[feature].to_numpy(dtype=float)
    shap_values = result.values[:, index]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.scatter(values, shap_values, s=12, alpha=0.35, color="#1f4e79", label="клиенты теста")
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 11)))
    centers: list[float] = []
    means: list[float] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (values >= low) & (values <= high)
        if not bool(mask.any()):
            continue
        centers.append(float(values[mask].mean()))
        means.append(float(shap_values[mask].mean()))
    if centers:
        ax.plot(centers, means, marker="o", color="#c00000", label="средний SHAP по децилям")
    ax.axhline(0.0, color="grey", linestyle=":")
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP (лог-шансы)")
    ax.set_title(f"Зависимость SHAP: {feature} ({result.label})")
    ax.legend(fontsize=8)
    return _save_figure(fig, path)


def plot_dependence_categorical(result: ShapResult, frame: pd.DataFrame, feature: str, path: Path) -> Path:
    """Для категориального признака — распределение SHAP по категориям (боксплоты, не scatter)."""
    index = result.features.index(feature)
    labels = frame[feature].astype(str)
    categories = sorted(labels.unique().tolist())
    data = [result.values[labels.to_numpy() == category, index] for category in categories]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    box = ax.boxplot(data, orientation="horizontal", showmeans=True, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#cfe3f7")
    ax.set_yticks(np.arange(1, len(categories) + 1))
    ax.set_yticklabels(categories)
    ax.axvline(0.0, color="grey", linestyle=":")
    ax.set_xlabel("SHAP (лог-шансы)")
    ax.set_ylabel(feature)
    ax.set_title(f"SHAP по категориям {feature} ({result.label})")
    return _save_figure(fig, path)


def pick_case(y_test: np.ndarray, proba: np.ndarray, threshold: float, seed: int = 42) -> int:
    """Клиент для waterfall: отток (y = 1) и вероятность не ниже порога модели (этап 6)."""
    candidates = np.where((np.asarray(y_test) == 1) & (np.asarray(proba) >= threshold))[0]
    if candidates.size == 0:
        raise RuntimeError(
            "нет тестовых клиентов с оттоком и вероятностью не ниже порога: waterfall построить нельзя"
        )
    return int(random.Random(seed).choice(candidates.tolist()))


def plot_waterfall_case(
    result: ShapResult,
    frame: pd.DataFrame,
    row: int,
    threshold: float,
    proba: float,
    path: Path,
) -> Path:
    """Разбор одного клиента: базовое значение и вклады 19 признаков (в лог-шансах)."""
    explanation = shap.Explanation(
        values=result.values[row],
        base_values=result.base_value,
        data=frame.iloc[row][result.features].to_numpy(dtype=object),
        feature_names=result.features,
    )
    shap.plots.waterfall(explanation, max_display=12, show=False)
    figure = plt.gcf()
    figure.set_size_inches(8.5, 5.5)
    plt.title(f"Разбор предсказания: {result.label}, p = {proba:.3f}, порог = {threshold:.2f}")
    plt.xlabel("вклад в лог-шансы (базовое значение + SHAP)")
    return _save_figure(figure, path)


def plot_spearman_heatmap(matrix: pd.DataFrame, path: Path) -> Path:
    """Матрица ρ Спирмена между важностями моделей (19 признаков)."""
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="ρ Спирмена")
    ax.set_title("Согласованность важностей: ρ Спирмена между моделями")
    return _save_figure(fig, path)


def make_figures(
    results: dict[str, ShapResult],
    matrix: pd.DataFrame,
    frame: pd.DataFrame,
    cfg: Config,
    case: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Фигуры этапа 8: bar и beeswarm по 5 моделям, dependence, waterfall и теплокарта ρ."""
    ensure_dir(cfg.figures_dir)
    keys = [key for key in MODEL_KEYS if key in results]
    paths: dict[str, Path] = {}
    for key in keys:
        result = results[key]
        paths[f"bar_{key}"] = plot_bar(result, cfg.figures_dir / f"shap_bar_{key}.png")
        paths[f"beeswarm_{key}"] = plot_beeswarm(result, cfg.figures_dir / f"shap_beeswarm_{key}.png")
    for key in DEPENDENCE_MODELS:
        if key not in results:
            continue
        result = results[key]
        for feature in DEPENDENCE_NUMERIC:
            paths[f"dependence_{feature}_{key}"] = plot_dependence_numeric(
                result, frame, feature, cfg.figures_dir / f"shap_dependence_{feature}_{key}.png"
            )
        paths[f"dependence_{DEPENDENCE_CATEGORICAL}_{key}"] = plot_dependence_categorical(
            result,
            frame,
            DEPENDENCE_CATEGORICAL,
            cfg.figures_dir / f"shap_dependence_{DEPENDENCE_CATEGORICAL}_{key}.png",
        )
    if case is not None and MAIN_MODEL_KEY in results:
        paths["waterfall"] = plot_waterfall_case(
            results[MAIN_MODEL_KEY],
            frame,
            int(case["row"]),
            float(case["threshold"]),
            float(case["proba"]),
            cfg.figures_dir / "shap_waterfall_case.png",
        )
    paths["spearman_heatmap"] = plot_spearman_heatmap(
        matrix, cfg.figures_dir / "shap_spearman_heatmap.png"
    )
    for name, path in paths.items():
        print(f"Сохранена фигура {name}: {path}")
    return paths


def save_table(table: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    table.to_csv(path, index=False)
    print(f"Сохранена таблица: {path}")
    return path


def print_frame(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Компактная печать нужных колонок без индекса."""
    print(frame.loc[:, list(columns)].to_string(index=False))


def print_report(
    importance: pd.DataFrame,
    top: pd.DataFrame,
    spearman: pd.DataFrame,
    matrix: pd.DataFrame,
    overlap: pd.DataFrame,
    interactions: dict[str, dict[str, Any]],
    check: pd.DataFrame,
    case: dict[str, Any] | None,
) -> None:
    """Печать этапа 8: топ-5 по моделям, матрица Спирмена, диагностика и краткий вывод."""
    print("Сверка с артефактом этапа 6 (test_predictions.parquet):")
    print_frame(check, ["key", "max_abs_diff", "atol", "ok"])
    print()
    for key in MODEL_KEYS:
        subset = top.loc[top["model"] == key].sort_values("rank", kind="stable")
        if subset.empty:
            continue
        print(f"{key}: топ-{TOP_N} признаков по среднему |SHAP| (лог-шансы)")
        print_frame(subset[["rank", "feature", "mean_abs_shap"]], ["rank", "feature", "mean_abs_shap"])
    print()
    print("ρ Спирмена между важностями моделей (19 исходных признаков):")
    print(matrix.round(3).to_string())
    rare = int((spearman["p_value"] < 0.05).sum())
    print(f"Пар с p < 0.05: {rare} из {len(spearman)}; минимальная ρ = {spearman['spearman'].min():.2f}")
    print()
    print("Пересечение топ-5 между парами моделей:")
    print_frame(overlap, ["model_a", "model_b", "n_overlap", "overlap_share", "common_features"])
    print()
    print("Диагностика аддитивности (shap_interaction_values, случайные строки теста, seed = 42):")
    for key, info in interactions.items():
        if info["supported"]:
            print(
                f"  {key:<18} строк = {info['rows']}, доля недиагональных вкладов = "
                f"{info['non_diagonal_share']:.2e}"
            )
        else:
            print(f"  {key:<18} пропущено: {info['reason']}")
    if case is not None:
        print(
            f"Waterfall: {MAIN_MODEL_KEY}, строка теста {case['row']}, p = {case['proba']:.3f}, "
            f"порог = {case['threshold']:.2f}"
        )
    print()
    leaders = importance.loc[importance["rank"] == 1].sort_values("mean_abs_shap", ascending=False)
    leader_text = ", ".join(f"{row.feature} ({row.model})" for row in leaders.itertuples())
    print("Вывод:")
    print(f"- Первое место по среднему |SHAP| у моделей: {leader_text}.")
    print(
        f"- Топ-5 моделей пересекаются в среднем на {overlap['n_overlap'].mean():.1f} из {TOP_N} признаков, "
        f"минимальная ρ вне диагонали = {spearman['spearman'].min():.2f}: состав важных признаков согласован."
    )
    print("- Значения SHAP — в лог-шансах, вклады OHE-столбцов сложены в 19 исходных признаков.")
    print(
        "Оговорка: TotalCharges, tenure и MonthlyCharges сильно коррелированы "
        "(TotalCharges ≈ tenure × MonthlyCharges), поэтому SHAP делит вклад между ними неоднозначно, "
        "и важность каждого признака в отдельности читается осторожно."
    )
    print("SHAP описывает модель, а не причинные связи.")


def run_explain(cfg: Config | None = None) -> dict[str, Path]:
    """Протокол этапа 8: обучение на train → сверка с артефактом → SHAP на тесте → таблицы и фигуры.

    Модели и параметры берутся из ``best_params.json`` без изменений; ни модель, ни порог здесь
    не выбираются. Тест читается один раз, только для описательного объяснения предсказаний.
    """
    configure_utf8_output()
    cfg = cfg or load_config()
    set_global_seed(cfg.seed)
    reset_test_reads()

    payload = read_best_params(cfg.tables_dir / BEST_PARAMS_FILE)
    specs = selected_specs(payload)
    keys = [spec.key for spec in specs]
    train = load_split(cfg)[0]
    X = train.drop(columns=[cfg.target])
    y = train[cfg.target]
    features = list(X.columns)
    if len(features) != EXPECTED_FEATURES:
        raise ValueError(f"ожидалось {EXPECTED_FEATURES} исходных признаков, получено {len(features)}")

    print("Этап 8: SHAP-интерпретация выбранных моделей (объясняем на тесте, выбор моделей не затрагивается)")
    print(f"Train: строк = {len(X)}, признаков = {len(features)}, доля оттока = {float(y.mean()):.4f}")
    print(f"Модели ({len(specs)}): {', '.join(keys)}")
    print(f"Параметры: best_params.json, число деревьев = median_best_iteration, seed = {cfg.seed}")

    thresholds = thresholds_from_json(cfg, specs)
    pipelines: dict[str, Any] = {}
    for spec in specs:
        pipeline, seconds = fit_on_train(spec, X, y, cfg)
        pipelines[spec.key] = pipeline
        print(f"  {spec.key:<18} обучена на всём train за {seconds:.1f} с")

    test = load_test_frame(cfg)
    X_test = test.drop(columns=[cfg.target])
    y_test = test[cfg.target]
    print(
        f"Тест: строк = {len(X_test)}, доля оттока = {float(y_test.mean()):.4f} — "
        "только для описательного объяснения предсказаний"
    )
    test_proba = {spec.key: predict_proba_test(pipelines[spec.key], X_test)[0] for spec in specs}
    check = check_predictions(cfg, specs, test_proba)
    print(
        "Сверка с test_predictions.parquet пройдена: предсказания совпали с этапом 6 "
        f"(atol = {ATOL_DEFAULT:g}, catboost = {ATOL_CATBOOST:g})"
    )
    print("SHAP-значения — в пространстве лог-шансов; OHE-столбцы суммируются в 19 исходных признаков")

    results: dict[str, ShapResult] = {}
    for spec in specs:
        result = explain_model(spec, pipelines[spec.key], X_test, features)
        results[spec.key] = result
        print(
            f"  {spec.key:<18} SHAP: форма {result.values.shape}, "
            f"базовое значение = {result.base_value:.4f}"
        )

    interactions = {
        spec.key: interaction_diagnostic(spec, pipelines[spec.key], X_test)
        for spec in specs
        if spec.key in INTERACTION_MODELS
    }
    importance = importance_frame([results[key] for key in keys], features)
    top = top_features_frame(importance)
    spearman = spearman_frame(importance, keys)
    matrix = spearman_matrix(spearman, keys)
    overlap = overlap_frame(top, keys)

    ensure_dir(cfg.tables_dir)
    paths = {
        "importance": save_table(importance, cfg.tables_dir / IMPORTANCE_FILE),
        "top_features": save_table(top, cfg.tables_dir / TOP_FEATURES_FILE),
        "spearman": save_table(spearman, cfg.tables_dir / SPEARMAN_FILE),
        "overlap": save_table(overlap, cfg.tables_dir / OVERLAP_FILE),
    }

    row = pick_case(y_test.to_numpy(), test_proba[MAIN_MODEL_KEY], thresholds[MAIN_MODEL_KEY])
    case = {
        "row": row,
        "threshold": thresholds[MAIN_MODEL_KEY],
        "proba": float(test_proba[MAIN_MODEL_KEY][row]),
    }
    paths.update(make_figures(results, matrix, X_test, cfg, case))

    print()
    print_report(importance, top, spearman, matrix, overlap, interactions, check, case)
    return paths


def main() -> None:
    """Точка входа `python -m src.explain` (цель `make explain`)."""
    paths = run_explain()
    print()
    print("Артефакты этапа 8:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()







