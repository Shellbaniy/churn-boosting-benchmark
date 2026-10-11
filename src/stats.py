"""Этап 7: статистическая значимость различий между tuned-моделями.

Гипотезы зафиксированы до расчёта (перебора пар в поисках значимых нет): 15 пар среди семи
tuned-моделей (ключи ``*_wide`` бустингов и ``logreg_A``):

- H1 — каждый tuned-бустинг против logreg_tuned (6 пар);
- H2 — вариант A против варианта B внутри алгоритма (3 пары);
- H3 — алгоритмы между собой внутри варианта A (3 пары) и внутри варианта B (3 пары).

Поправка Holm применяется внутри семейства и отдельно для каждой метрики (m = число пар
семейства: 6, 3 и 6).

Три независимые части:
1. CV на train: RepeatedStratifiedKFold(cv_splits × cv_repeats) даёт одни и те же разбиения
   для всех моделей; попарный corrected resampled t-test (Nadeau & Bengio, 2003; схема
   повторной k-fold — Bouckaert & Frank, 2004), d_z, доверительный интервал среднего
   различия и минимально обнаружимое различие (MDE).
2. Bootstrap на тесте (B выборок строк): ДИ ROC-AUC, PR-AUC и прибыли на клиента плюс
   парные разности с бутстрэп-p. Читается только ``reports/tables/test_predictions.parquet``
   этапа 6.
3. McNemar exact на тесте при порогах из ``reports/tables/thresholds.json``.

Тест повторно не читается: модуль не обращается к тестовой части split и не вызывает функцию
единственного чтения теста из ``src.evaluate``; тестовые числа берутся из сохранённых артефактов
этапа 6.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # фигуры сохраняются в файлы, интерактивный backend не нужен

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402

from src.config import BusinessConfig, Config, load_business_config, load_config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.evaluate import (  # noqa: E402
    final_best_keys,
    positive_class_column,
    profit_at_threshold,
    read_best_params,
    tuned_params,
)
from src.pipelines import MODEL_LABELS, build_pipeline  # noqa: E402
from src.utils import configure_utf8_output, ensure_dir, read_json, set_global_seed  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

BEST_PARAMS_FILE = "best_params.json"
THRESHOLDS_FILE = "thresholds.json"
TEST_PREDICTIONS_FILE = "test_predictions.parquet"
CV_SCORES_FILE = "stats_cv_scores.csv"
TTEST_FILE = "stats_resampled_ttest.csv"
BOOTSTRAP_CI_FILE = "stats_bootstrap_ci.csv"
BOOTSTRAP_DIFF_FILE = "stats_bootstrap_diff.csv"
MCNEMAR_FILE = "stats_mcnemar_holm.csv"
FIGURE_FILES = {
    "auc_ci": "stats_auc_ci.png",
    "cv_boxplot": "stats_cv_boxplot.png",
    "pvalue_heatmap": "stats_pvalue_heatmap.png",
}
TTEST_METRICS: tuple[str, ...] = ("roc_auc", "pr_auc")
BOOTSTRAP_METRICS: tuple[str, ...] = ("roc_auc", "pr_auc", "profit_per_customer")
METRIC_LABELS = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "profit_per_customer": "прибыль на клиента"}
ALPHA = 0.05
N_BOOTSTRAP = 5000
FAMILY_H1 = "H1: бустинг vs logreg_tuned"
FAMILY_H2 = "H2: вариант A vs B"
FAMILY_H3 = "H3: алгоритм vs алгоритм"
FAMILIES: tuple[str, ...] = (FAMILY_H1, FAMILY_H2, FAMILY_H3)
ALGORITHMS: tuple[str, ...] = ("xgb", "lgbm", "catboost")


@dataclass(frozen=True)
class TunedModel:
    """Одна tuned-модель: ключ в best_params.json (CV) и отображение на ключ этапа 6."""

    key: str
    model_name: str
    variant: str

    @property
    def label(self) -> str:
        return f"{MODEL_LABELS[self.model_name]} {self.variant}"

    @property
    def evaluate_key(self) -> str:
        """Ключ модели на этапе 6: колонка ``p_<ключ>`` в test_predictions.parquet и запись в thresholds.json.

        Ключи best_params.json ``*_wide``/``logreg_A`` и ключи бенчмарка ``*_tuned``/``logreg_tuned``
        различаются, поэтому отображение задаётся в одном месте: CV и гиперпараметры берутся по
        ``key``, тестовые предсказания и пороги — по ``evaluate_key``.
        """
        if self.model_name == "logreg":
            return "logreg_tuned"
        return f"{self.model_name}_{self.variant}_tuned"


@dataclass(frozen=True)
class Pair:
    """Одна заранее зарегистрированная пара: разность считается как score(a) - score(b)."""

    family: str
    a: str
    b: str

    @property
    def name(self) -> str:
        return f"{self.a} vs {self.b}"


def tuned_models() -> list[TunedModel]:
    """Семь tuned-моделей те же, что в бенчмарке: ключи берутся из final_best_keys()."""
    models: list[TunedModel] = []
    for key in final_best_keys():
        parts = key.split("_")
        if len(parts) < 2 or parts[0] not in MODEL_LABELS:
            raise ValueError(f"Ключ {key!r} не разбирается на модель и вариант препроцессинга.")
        models.append(TunedModel(key=key, model_name=parts[0], variant=parts[1]))
    return models


def preregistered_pairs(models: Sequence[TunedModel]) -> list[Pair]:
    """15 пар трёх семейств; состав зафиксирован до просмотра результатов."""
    by_name = {(model.model_name, model.variant): model.key for model in models}
    logreg = by_name[("logreg", "A")]
    boosters = [model.key for model in models if model.model_name != "logreg"]
    pairs = [Pair(FAMILY_H1, key, logreg) for key in boosters]
    for algorithm in ALGORITHMS:
        pairs.append(Pair(FAMILY_H2, by_name[(algorithm, "A")], by_name[(algorithm, "B")]))
    for variant in ("A", "B"):
        for left, right in (("xgb", "lgbm"), ("xgb", "catboost"), ("lgbm", "catboost")):
            pairs.append(Pair(FAMILY_H3, by_name[(left, variant)], by_name[(right, variant)]))
    unique = {(pair.a, pair.b) for pair in pairs}
    if len(pairs) != 15 or len(unique) != 15:
        raise ValueError(f"Зарегистрировано {len(pairs)} пар, уникальных {len(unique)}, ожидается 15.")
    return pairs


def holm_correction(p_values: Iterable[float]) -> np.ndarray:
    """Поправка Holm внутри семейства: монотонные скорректированные p в исходном порядке."""
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return values
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"p-value вне [0, 1]: {values.tolist()}")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(values.size, dtype=float)
    running = 0.0
    total = values.size
    for rank, index in enumerate(order, start=1):
        scaled = min(1.0, float(values[index]) * (total - rank + 1))
        running = max(running, scaled)
        adjusted[index] = running
    return adjusted


def add_holm(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    """Дописывает ``p_holm``: Holm отдельно внутри каждой группы (семейство × метрика)."""
    if "p" not in frame.columns:
        raise ValueError("Нет колонки 'p': поправка Holm применяется к исходным p-value.")
    result = frame.copy()
    result["p_holm"] = np.nan
    for _, group in result.groupby(list(group_columns), sort=False):
        result.loc[group.index, "p_holm"] = holm_correction(group["p"])
    return result


def corrected_resampled_ttest(
    differences: np.ndarray,
    ratio_test_train: float,
    alpha: float = ALPHA,
) -> dict[str, float]:
    """Corrected resampled t-test для повторной k-fold (Nadeau & Bengio, 2003).

    ``t = mean(d) / sqrt((1/k + n_test/n_train) * var(d, ddof=1))``, df = k - 1, двусторонний p.
    Множитель ``n_test/n_train`` учитывает перекрытие обучающих частей фолдов и делает тест
    консервативнее обычного парного t-теста по фолдам. Дополнительно возвращаются d_z
    (Cohen's d для парных наблюдений) и доверительный интервал среднего различия.
    """
    diff = np.asarray(differences, dtype=float)
    if diff.ndim != 1 or diff.size < 2:
        raise ValueError(f"нужен одномерный массив минимум из двух различий, получено {diff.shape}")
    if not np.all(np.isfinite(diff)):
        raise ValueError("в различиях есть NaN/inf: пары не сошлись по фолдам")
    k = int(diff.size)
    mean_diff = float(diff.mean())
    variance = float(diff.var(ddof=1))
    std_diff = float(np.sqrt(variance))
    df = k - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    standard_error = float(np.sqrt((1.0 / k + float(ratio_test_train)) * variance))
    if standard_error > 0.0:
        t_stat = mean_diff / standard_error
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df))
        d_z = mean_diff / std_diff
    else:
        # все различия одинаковы: t-статистика не определена, решение по знаку различия
        t_stat = float("inf") if mean_diff > 0 else (float("-inf") if mean_diff < 0 else 0.0)
        p_value = 0.0 if mean_diff != 0.0 else 1.0
        d_z = 0.0
    half_width = t_crit * standard_error
    return {
        "mean_diff": mean_diff,
        "se": standard_error,
        "t": t_stat,
        "df": float(df),
        "p": p_value,
        "d_z": float(d_z),
        "ci_low": mean_diff - half_width,
        "ci_high": mean_diff + half_width,
        "t_crit": t_crit,
        "n_folds": float(k),
        "ratio_test_train": float(ratio_test_train),
    }


def minimal_detectable_difference(
    table: pd.DataFrame,
    metric: str = "roc_auc",
    alpha: float = ALPHA,
) -> dict[str, float]:
    """MDE = t_crit(1 - alpha/2, k - 1) × медианный SE по парам.

    Различия меньше MDE при данной схеме CV статистически неразличимы при выбранном alpha;
    медиана SE берётся по всем парам метрики.
    """
    rows = table.loc[table["metric"] == metric]
    if rows.empty:
        raise ValueError(f"в таблице нет строк для метрики {metric!r}")
    df = int(rows["df"].iloc[0])
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    median_se = float(rows["se"].median())
    return {"metric": metric, "df": df, "t_crit": t_crit, "median_se": median_se, "mde": t_crit * median_se}


def score_series(frame: pd.DataFrame, metric: str) -> dict[str, np.ndarray]:
    """Оценки по разбиениям в едином порядке (repeat, fold) для каждой модели."""
    if metric not in frame.columns:
        raise KeyError(f"в таблице CV-оценок нет колонки {metric!r}: {list(frame.columns)}")
    ordered = frame.sort_values(["model", "repeat", "fold"], kind="stable")
    return {str(key): group[metric].to_numpy(dtype=float) for key, group in ordered.groupby("model", sort=True)}


def repeated_folds(X: pd.DataFrame, y: pd.Series, cfg: Config) -> list[tuple[np.ndarray, np.ndarray]]:
    """Разбиения CV train: cv_splits × cv_repeats индексов, одинаковые для всех моделей."""
    cv = RepeatedStratifiedKFold(
        n_splits=cfg.cv_splits,
        n_repeats=cfg.cv_repeats,
        random_state=cfg.seed,
    )
    return list(cv.split(X, y))


def fold_ratio(folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    """Среднее ``n_test/n_train`` по фолдам — множитель поправки Nadeau & Bengio."""
    if not folds:
        raise ValueError("список разбиений пуст")
    return float(np.mean([valid.size / train.size for train, valid in folds]))


def cv_score_frame(
    models: Sequence[TunedModel],
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Config,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    payload: dict[str, Any],
) -> pd.DataFrame:
    """ROC-AUC и PR-AUC каждой tuned-модели на каждом разбиении (long-формат)."""
    if not folds:
        raise ValueError("нужен непустой список разбиений CV")
    rows: list[dict[str, Any]] = []
    total = len(folds)
    for model in models:
        if model.key not in payload:
            raise KeyError(f"{model.key} отсутствует в best_params.json: пары настроить нечем.")
        params = tuned_params(model.model_name, payload[model.key])
        scores: list[float] = []
        for position, (train_idx, valid_idx) in enumerate(folds):
            pipeline = build_pipeline(model.model_name, model.variant, params, cfg)
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba_all = pipeline.predict_proba(X.iloc[valid_idx])
            proba = proba_all[:, positive_class_column(pipeline)]
            y_valid = y.iloc[valid_idx].to_numpy()
            roc_auc = float(roc_auc_score(y_valid, proba))
            pr_auc = float(average_precision_score(y_valid, proba))
            scores.append(roc_auc)
            rows.append(
                {
                    "model": model.key,
                    "repeat": position // cfg.cv_splits,
                    "fold": position % cfg.cv_splits,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                }
            )
            print(
                f"  {model.key:<18} разбиение {position + 1}/{total}: "
                f"ROC-AUC = {roc_auc:.4f}, PR-AUC = {pr_auc:.4f}",
                flush=True,
            )
        print(f"  {model.key:<18} среднее по {total} разбиениям: ROC-AUC = {np.mean(scores):.4f}", flush=True)
    frame = pd.DataFrame(rows, columns=["model", "repeat", "fold", "roc_auc", "pr_auc"])
    return frame.round({"roc_auc": 6, "pr_auc": 6})


def load_cv_scores(models: Sequence[TunedModel], cfg: Config) -> pd.DataFrame | None:
    """Кэш CV-оценок, если он полный и согласован с текущим набором моделей."""
    path = cfg.tables_dir / CV_SCORES_FILE
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {"model", "repeat", "fold", "roc_auc", "pr_auc"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Кэш {path} повреждён: нет колонок {sorted(required - set(frame.columns))}")
    expected_keys = {model.key for model in models}
    actual_keys = set(frame["model"].astype(str))
    if actual_keys != expected_keys:
        raise ValueError(f"Кэш {path} собран для других моделей: {sorted(actual_keys)} != {sorted(expected_keys)}")
    expected_rows = len(models) * cfg.cv_splits * cfg.cv_repeats
    if len(frame) != expected_rows:
        raise ValueError(f"Кэш {path}: строк {len(frame)}, ожидается {expected_rows}; запустите с --force.")
    return frame


def cv_scores(
    models: Sequence[TunedModel],
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Config,
    payload: dict[str, Any],
    force: bool = False,
) -> tuple[pd.DataFrame, float]:
    """CV-оценки с кэшем: ``stats_cv_scores.csv`` пересчитывается только с ``--force``."""
    folds = repeated_folds(X, y, cfg)
    ratio = fold_ratio(folds)
    if not force:
        cached = load_cv_scores(models, cfg)
        if cached is not None:
            print(f"CV-оценки уже посчитаны: {cfg.tables_dir / CV_SCORES_FILE} (строк = {len(cached)}).")
            print("Пересчёт — только с флагом --force.")
            return cached, ratio
    print(
        f"CV на train: RepeatedStratifiedKFold(n_splits = {cfg.cv_splits}, "
        f"n_repeats = {cfg.cv_repeats}, seed = {cfg.seed}) → разбиений {len(folds)}, "
        f"моделей {len(models)}"
    )
    print(f"Среднее n_test/n_train по разбиениям (поправка Nadeau & Bengio): {ratio:.4f}")
    frame = cv_score_frame(models, X, y, cfg, folds, payload)
    path = cfg.tables_dir / CV_SCORES_FILE
    ensure_dir(cfg.tables_dir)
    frame.to_csv(path, index=False)
    print(f"Сохранены CV-оценки: {path}")
    return frame, ratio


def ttest_frame(
    frame: pd.DataFrame,
    pairs: Sequence[Pair],
    models: Sequence[TunedModel],
    ratio_test_train: float,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Таблица corrected resampled t-test по всем парам для ROC-AUC и PR-AUC."""
    labels = {model.key: model.label for model in models}
    evaluate_keys = {model.key: model.evaluate_key for model in models}
    rows: list[dict[str, Any]] = []
    for metric in TTEST_METRICS:
        series = score_series(frame, metric)
        missing = [pair.name for pair in pairs if pair.a not in series or pair.b not in series]
        if missing:
            raise KeyError(f"нет CV-оценок для пар: {missing}")
        for pair in pairs:
            result = corrected_resampled_ttest(series[pair.a] - series[pair.b], ratio_test_train, alpha)
            rows.append(
                {
                    "pair": pair.name,
                    "family": pair.family,
                    "metric": metric,
                    "model_a": pair.a,
                    "model_b": pair.b,
                    "label_a": labels[pair.a],
                    "label_b": labels[pair.b],
                    "evaluate_key_a": evaluate_keys[pair.a],
                    "evaluate_key_b": evaluate_keys[pair.b],
                    "mean_a": float(series[pair.a].mean()),
                    "mean_b": float(series[pair.b].mean()),
                    **result,
                }
            )
    ordered = [
        "pair",
        "family",
        "metric",
        "model_a",
        "model_b",
        "label_a",
        "label_b",
        "evaluate_key_a",
        "evaluate_key_b",
        "mean_a",
        "mean_b",
    ]
    table = add_holm(pd.DataFrame(rows), ("metric", "family"))
    table = table[ordered + [column for column in table.columns if column not in ordered]]
    rounding = {
        "mean_a": 6,
        "mean_b": 6,
        "mean_diff": 6,
        "se": 6,
        "t": 4,
        "p": 8,
        "p_holm": 8,
        "d_z": 4,
        "ci_low": 6,
        "ci_high": 6,
        "t_crit": 4,
        "ratio_test_train": 4,
    }
    return table.round(rounding)


def probability_column(frame: pd.DataFrame, evaluate_key: str) -> str:
    """Колонка вероятностей модели в ``test_predictions.parquet`` (этап 6 пишет ``p_<ключ>``)."""
    for candidate in (f"p_{evaluate_key}", evaluate_key):
        if candidate in frame.columns:
            return candidate
    raise KeyError(
        f"в test_predictions.parquet нет колонки вероятностей для {evaluate_key!r}; "
        f"доступные колонки: {list(frame.columns)}"
    )


def load_test_predictions(cfg: Config, models: Sequence[TunedModel]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Метки теста и вероятности моделей из артефакта этапа 6 (сам тест не читается)."""
    path = cfg.tables_dir / TEST_PREDICTIONS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла предсказаний на тесте: {path}. Сначала выполните `make evaluate` (этап 6)."
        )
    frame = pd.read_parquet(path)
    if "y_true" not in frame.columns:
        raise KeyError(f"в {path} нет колонки y_true; доступные колонки: {list(frame.columns)}")
    y_true = frame["y_true"].to_numpy(dtype=float)
    proba = {
        model.key: frame[probability_column(frame, model.evaluate_key)].to_numpy(dtype=float)
        for model in models
    }
    return y_true, proba


def thresholds_from_json(cfg: Config, models: Sequence[TunedModel]) -> dict[str, float]:
    """Пороги моделей из ``thresholds.json`` (этап 6): подобраны на OOF train.

    Возвращаемый словарь ключуется ключами best_params.json (``model.key``): внутри этапа 7 так
    адресуются предсказания, а в файле пороги лежат под ключами бенчмарка (``model.evaluate_key``).
    """
    path = cfg.tables_dir / THRESHOLDS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Нет файла порогов: {path}. Сначала выполните `make evaluate` (этап 6).")
    payload = read_json(path)
    entries = payload.get("thresholds", payload)
    thresholds: dict[str, float] = {}
    for model in models:
        if model.evaluate_key not in entries:
            raise KeyError(
                f"в {path} нет порога для {model.evaluate_key!r}; "
                f"доступные записи: {sorted(entries)}"
            )
        entry = entries[model.evaluate_key]
        value = entry["threshold"] if isinstance(entry, dict) else entry
        thresholds[model.key] = float(value)
    return thresholds


def bootstrap_distributions(
    y_true: np.ndarray,
    proba: dict[str, np.ndarray],
    thresholds: dict[str, float],
    business: BusinessConfig,
    keys: Sequence[str],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 42,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Бутстрэп-распределения ROC-AUC, PR-AUC и прибыли на клиента.

    Индексы строк разыгрываются один раз на итерацию и используются для всех моделей, поэтому
    разности по парам получаются парными (одни и те же клиенты у обеих моделей). Итерации, где
    в выборке остался один класс, пропускаются: ROC-AUC на них не определён.
    """
    size = int(y_true.size)
    if size == 0:
        raise ValueError("пустая тестовая выборка")
    rng = np.random.default_rng(seed)
    distributions: dict[str, dict[str, np.ndarray]] = {
        key: {name: np.empty(n_bootstrap, dtype=float) for name in BOOTSTRAP_METRICS} for key in keys
    }
    kept = 0
    skipped = 0
    for _ in range(n_bootstrap):
        index = rng.integers(0, size, size=size)
        y_boot = y_true[index]
        if np.unique(y_boot).size < 2:
            skipped += 1
            continue
        for key in keys:
            proba_boot = proba[key][index]
            distributions[key]["roc_auc"][kept] = roc_auc_score(y_boot, proba_boot)
            distributions[key]["pr_auc"][kept] = average_precision_score(y_boot, proba_boot)
            distributions[key]["profit_per_customer"][kept] = (
                profit_at_threshold(
                    y_boot,
                    proba_boot,
                    thresholds[key],
                    business.customer_lifetime_value,
                    business.retention_offer_cost,
                    business.retention_success_rate,
                )
                / size
            )
        kept += 1
    for key in keys:
        for name in BOOTSTRAP_METRICS:
            distributions[key][name] = distributions[key][name][:kept]
    meta = {"n_bootstrap": kept, "skipped": skipped, "seed": seed, "n_test": size}
    return distributions, meta
def bootstrap_p_value(differences: np.ndarray) -> float:
    """Двусторонний бутстрэп-p: ``2 * min(P(d <= 0), P(d >= 0))``, но не меньше ``1/B``.

    Ноль попадает в обе группы, поэтому для вырожденного распределения (все разности равны нулю)
    p = 1. Нижняя граница ``1/B`` — разрешение бутстрэпа: p = 0 означало бы бесконечную точность.
    """
    diff = np.asarray(differences, dtype=float)
    if diff.size == 0:
        raise ValueError("пустое распределение разностей")
    below = float((diff <= 0.0).mean())
    above = float((diff >= 0.0).mean())
    return float(min(1.0, max(2.0 * min(below, above), 1.0 / diff.size)))


def bootstrap_ci_frame(
    y_true: np.ndarray,
    proba: dict[str, np.ndarray],
    thresholds: dict[str, float],
    business: BusinessConfig,
    distributions: dict[str, dict[str, np.ndarray]],
    models: Sequence[TunedModel],
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Точечные значения и перцентильные ДИ на тесте по каждой модели и метрике."""
    lo, hi = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    size = int(y_true.size)
    rows: list[dict[str, Any]] = []
    for model in models:
        observed = {
            "roc_auc": float(roc_auc_score(y_true, proba[model.key])),
            "pr_auc": float(average_precision_score(y_true, proba[model.key])),
            "profit_per_customer": profit_at_threshold(
                y_true,
                proba[model.key],
                thresholds[model.key],
                business.customer_lifetime_value,
                business.retention_offer_cost,
                business.retention_success_rate,
            )
            / size,
        }
        for metric in BOOTSTRAP_METRICS:
            sample = distributions[model.key][metric]
            rows.append(
                {
                    "model": model.key,
                    "evaluate_key": model.evaluate_key,
                    "label": model.label,
                    "metric": metric,
                    "value": observed[metric],
                    "ci_low": float(np.percentile(sample, lo)),
                    "ci_high": float(np.percentile(sample, hi)),
                    "threshold": float(thresholds[model.key]),
                }
            )
    return pd.DataFrame(rows).round(
        {"value": 6, "ci_low": 6, "ci_high": 6, "threshold": 4}
    )


def bootstrap_diff_frame(
    distributions: dict[str, dict[str, np.ndarray]],
    pairs: Sequence[Pair],
    models: Sequence[TunedModel],
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Парные разности на тесте: среднее, перцентильный ДИ, бутстрэп-p и p_holm."""
    labels = {model.key: model.label for model in models}
    evaluate_keys = {model.key: model.evaluate_key for model in models}
    lo, hi = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        for metric in BOOTSTRAP_METRICS:
            differences = distributions[pair.a][metric] - distributions[pair.b][metric]
            rows.append(
                {
                    "pair": pair.name,
                    "family": pair.family,
                    "metric": metric,
                    "model_a": pair.a,
                    "model_b": pair.b,
                    "label_a": labels[pair.a],
                    "label_b": labels[pair.b],
                    "evaluate_key_a": evaluate_keys[pair.a],
                    "evaluate_key_b": evaluate_keys[pair.b],
                    "diff_mean": float(differences.mean()),
                    "ci_low": float(np.percentile(differences, lo)),
                    "ci_high": float(np.percentile(differences, hi)),
                    "p": bootstrap_p_value(differences),
                }
            )
    table = add_holm(pd.DataFrame(rows), ("metric", "family"))
    ordering = [
        "pair",
        "family",
        "metric",
        "model_a",
        "model_b",
        "label_a",
        "label_b",
        "evaluate_key_a",
        "evaluate_key_b",
    ]
    table = table[ordering + [column for column in table.columns if column not in ordering]]
    return table.round({"diff_mean": 6, "ci_low": 6, "ci_high": 6, "p": 8, "p_holm": 8})


def mcnemar_exact_p(b_discordant: int, c_discordant: int) -> float:
    """Точный двусторонний McNemar: биномиальный тест b из (b + c) при p = 0.5."""
    total = int(b_discordant) + int(c_discordant)
    if total == 0:
        return 1.0
    return float(stats.binomtest(int(b_discordant), total, 0.5).pvalue)


def mcnemar_frame(
    y_true: np.ndarray,
    proba: dict[str, np.ndarray],
    thresholds: dict[str, float],
    pairs: Sequence[Pair],
    models: Sequence[TunedModel],
) -> pd.DataFrame:
    """McNemar exact по каждой паре при порогах из thresholds.json.

    ``b`` — модель A права, B ошибается; ``c`` — наоборот. Дискордантные пары (b + c) задают
    разрешение теста: при малом их числе точный тест почти ничего не различает.
    """
    labels = {model.key: model.label for model in models}
    evaluate_keys = {model.key: model.evaluate_key for model in models}
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        correct_a = (proba[pair.a] >= thresholds[pair.a]) == y_true
        correct_b = (proba[pair.b] >= thresholds[pair.b]) == y_true
        b_discordant = int(np.sum(correct_a & ~correct_b))
        c_discordant = int(np.sum(~correct_a & correct_b))
        rows.append(
            {
                "pair": pair.name,
                "family": pair.family,
                "model_a": pair.a,
                "model_b": pair.b,
                "label_a": labels[pair.a],
                "label_b": labels[pair.b],
                "evaluate_key_a": evaluate_keys[pair.a],
                "evaluate_key_b": evaluate_keys[pair.b],
                "threshold_a": float(thresholds[pair.a]),
                "threshold_b": float(thresholds[pair.b]),
                "b_a_right": b_discordant,
                "c_b_right": c_discordant,
                "n_discordant": b_discordant + c_discordant,
                "p": mcnemar_exact_p(b_discordant, c_discordant),
            }
        )
    table = add_holm(pd.DataFrame(rows), ("family",))
    ordering = [
        "pair",
        "family",
        "model_a",
        "model_b",
        "label_a",
        "label_b",
        "evaluate_key_a",
        "evaluate_key_b",
    ]
    table = table[ordering + [column for column in table.columns if column not in ordering]]
    return table.round({"threshold_a": 4, "threshold_b": 4, "p": 8, "p_holm": 8})


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cv_boxplot(frame: pd.DataFrame, models: Sequence[TunedModel], path: Path) -> Path:
    """Разброс ROC-AUC по разбиениям CV: видно, что различие между моделями меньше разброса."""
    labels = [model.label for model in models]
    data = [frame.loc[frame["model"] == model.key, "roc_auc"].to_numpy(dtype=float) for model in models]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    box = ax.boxplot(data, orientation="horizontal", showmeans=True, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#cfe3f7")
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.set_xlabel("ROC-AUC на разбиениях CV")
    ax.set_title(f"ROC-AUC по {frame['repeat'].nunique() * frame['fold'].nunique()} разбиениям CV на train")
    ax.grid(axis="x", alpha=0.3)
    return _save_figure(fig, path)


def plot_auc_ci(ci_table: pd.DataFrame, path: Path, n_bootstrap: int) -> Path:
    """Точечные оценки ROC-AUC на тесте с бутстрэп-интервалами (forest plot)."""
    rows = ci_table.loc[ci_table["metric"] == "roc_auc"].sort_values("value")
    positions = np.arange(len(rows))
    errors = np.vstack([rows["value"] - rows["ci_low"], rows["ci_high"] - rows["value"]])
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.errorbar(rows["value"], positions, xerr=errors, fmt="o", color="#1f4e79", capsize=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(rows["label"])
    ax.set_xlabel("ROC-AUC на тесте (95% bootstrap CI)")
    ax.set_title(f"ROC-AUC на тесте: бутстрэп-интервалы (B = {n_bootstrap})")
    ax.grid(axis="x", alpha=0.3)
    return _save_figure(fig, path)


def plot_pvalue_heatmap(table: pd.DataFrame, models: Sequence[TunedModel], path: Path) -> Path:
    """Матрица p_holm corrected resampled t-test (ROC-AUC); зарегистрированные пары заполнены."""
    keys = [model.key for model in models]
    index = {key: position for position, key in enumerate(keys)}
    matrix = np.full((len(keys), len(keys)), np.nan, dtype=float)
    rows = table.loc[table["metric"] == "roc_auc"]
    for row in rows.itertuples():
        i, j = index[row.model_a], index[row.model_b]
        matrix[i, j] = matrix[j, i] = float(row.p_holm)
    cmap = matplotlib.colormaps["YlOrRd"].with_extremes(bad="#d9d9d9")
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels(keys)
    for i in range(len(keys)):
        for j in range(len(keys)):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            ax.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="p_holm (ROC-AUC)")
    ax.set_title("p_holm corrected resampled t-test (ROC-AUC)")
    return _save_figure(fig, path)


def make_figures(
    cv_frame: pd.DataFrame,
    ci_table: pd.DataFrame,
    ttest_table: pd.DataFrame,
    models: Sequence[TunedModel],
    figures_dir: Path,
    n_bootstrap: int,
) -> dict[str, Path]:
    """Три фигуры этапа 7: разброс по CV, интервалы на тесте, матрица p_holm."""
    paths = {
        "cv_boxplot": plot_cv_boxplot(cv_frame, models, figures_dir / FIGURE_FILES["cv_boxplot"]),
        "auc_ci": plot_auc_ci(ci_table, figures_dir / FIGURE_FILES["auc_ci"], n_bootstrap),
        "pvalue_heatmap": plot_pvalue_heatmap(ttest_table, models, figures_dir / FIGURE_FILES["pvalue_heatmap"]),
    }
    for name, figure_path in paths.items():
        print(f"Сохранена фигура {name}: {figure_path}")
    return paths


def _format_value(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    number = float(value)
    if not np.isfinite(number):
        return "-"
    if number != 0.0 and abs(number) < 10.0 ** (-decimals):
        return f"{number:.2e}"
    return f"{number:.{decimals}f}"


def print_frame(frame: pd.DataFrame, columns: Sequence[str], decimals: int = 4) -> None:
    view = frame.loc[:, list(columns)]
    formatted = pd.DataFrame(
        {column: [_format_value(value, decimals) for value in view[column]] for column in view.columns}
    )
    print(formatted.to_string(index=False))


def save_table(table: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    table.to_csv(path, index=False)
    print(f"Сохранена таблица: {path}")
    return path


def print_cv_summary(cv_frame: pd.DataFrame, models: Sequence[TunedModel], ratio_test_train: float) -> None:
    rows = []
    for model in models:
        subset = cv_frame.loc[cv_frame["model"] == model.key]
        rows.append(
            {
                "model": model.key,
                "roc_auc_mean": float(subset["roc_auc"].mean()),
                "roc_auc_std": float(subset["roc_auc"].std(ddof=1)),
                "roc_auc_min": float(subset["roc_auc"].min()),
                "roc_auc_max": float(subset["roc_auc"].max()),
                "pr_auc_mean": float(subset["pr_auc"].mean()),
            }
        )
    print(f"CV-оценки по моделям (n_test/n_train = {ratio_test_train:.4f}):")
    print_frame(
        pd.DataFrame(rows),
        ["model", "roc_auc_mean", "roc_auc_std", "roc_auc_min", "roc_auc_max", "pr_auc_mean"],
    )


def print_ttest_report(table: pd.DataFrame, alpha: float) -> None:
    columns = [
        "pair",
        "label_a",
        "label_b",
        "mean_a",
        "mean_b",
        "mean_diff",
        "ci_low",
        "ci_high",
        "se",
        "t",
        "p",
        "p_holm",
        "d_z",
    ]
    for metric in TTEST_METRICS:
        for family in FAMILIES:
            subset = table.loc[(table["metric"] == metric) & (table["family"] == family)]
            if subset.empty:
                continue
            print(f"{METRIC_LABELS[metric]}, семейство {family} (поправка Holm внутри семейства):")
            print_frame(subset, columns)
    for metric in TTEST_METRICS:
        subset = table.loc[table["metric"] == metric]
        significant = int((subset["p_holm"] < alpha).sum())
        print(
            f"{METRIC_LABELS[metric]}: значимых пар при alpha = {alpha} после Holm — "
            f"{significant} из {len(subset)}"
        )


def print_mde(mde: dict[str, float], alpha: float) -> None:
    print(
        f"MDE ({METRIC_LABELS[mde['metric']]}, df = {mde['df']:.0f}, "
        f"t_crit = {mde['t_crit']:.3f}, медианный SE = {mde['median_se']:.4f}): {mde['mde']:.4f}"
    )
    print(
        f"Различия {METRIC_LABELS[mde['metric']]} меньше {mde['mde']:.4f} при {int(mde['df']) + 1} "
        f"разбиениях CV и alpha = {alpha} статистически неразличимы этой схемой."
    )


def print_bootstrap_report(
    ci_table: pd.DataFrame,
    diff_table: pd.DataFrame,
    meta: dict[str, Any],
    alpha: float,
) -> None:
    print(
        f"Бутстрэп на тесте: строк {meta['n_test']}, выборок использовано {meta['n_bootstrap']} "
        f"(пропущено {meta['skipped']}), seed = {meta['seed']}"
    )
    for metric in BOOTSTRAP_METRICS:
        subset = ci_table.loc[ci_table["metric"] == metric].sort_values("value", ascending=False)
        print(f"{METRIC_LABELS[metric]}: точечная оценка и 95% интервал")
        print_frame(subset, ["evaluate_key", "label", "value", "ci_low", "ci_high", "threshold"])
    columns = ["pair", "label_a", "label_b", "diff_mean", "ci_low", "ci_high", "p", "p_holm"]
    for metric in BOOTSTRAP_METRICS:
        for family in FAMILIES:
            subset = diff_table.loc[(diff_table["metric"] == metric) & (diff_table["family"] == family)]
            if subset.empty:
                continue
            print(f"{METRIC_LABELS[metric]}, семейство {family}: разность A - B на тесте")
            print_frame(subset, columns)
    for metric in BOOTSTRAP_METRICS:
        subset = diff_table.loc[diff_table["metric"] == metric]
        significant = int((subset["p_holm"] < alpha).sum())
        print(
            f"{METRIC_LABELS[metric]}: значимых разностей при alpha = {alpha} после Holm — "
            f"{significant} из {len(subset)}"
        )


def print_mcnemar_report(table: pd.DataFrame, alpha: float) -> None:
    columns = [
        "pair",
        "label_a",
        "label_b",
        "threshold_a",
        "threshold_b",
        "b_a_right",
        "c_b_right",
        "n_discordant",
        "p",
        "p_holm",
    ]
    for family in FAMILIES:
        subset = table.loc[table["family"] == family]
        if subset.empty:
            continue
        print(f"McNemar exact, семейство {family} (пороги из thresholds.json):")
        print_frame(subset, columns)
    significant = int((table["p_holm"] < alpha).sum())
    print(
        f"McNemar exact: значимых пар при alpha = {alpha} после Holm — {significant} из {len(table)}; "
        f"медиана дискордантных пар — {float(table['n_discordant'].median()):.0f}"
    )


def print_conclusion(
    ttest_table: pd.DataFrame,
    diff_table: pd.DataFrame,
    mcnemar_table: pd.DataFrame,
    mde: dict[str, float],
    alpha: float,
) -> None:
    print("Итог по трём тестам на зарегистрированных парах:")
    for metric in TTEST_METRICS:
        subset = ttest_table.loc[ttest_table["metric"] == metric]
        print(
            f"- CV corrected t-test, {METRIC_LABELS[metric]}: значимых "
            f"{int((subset['p_holm'] < alpha).sum())} из {len(subset)}"
        )
    for metric in BOOTSTRAP_METRICS:
        subset = diff_table.loc[diff_table["metric"] == metric]
        print(
            f"- Бутстрэп на тесте, {METRIC_LABELS[metric]}: значимых "
            f"{int((subset['p_holm'] < alpha).sum())} из {len(subset)}"
        )
    print(
        f"- McNemar exact на тесте: значимых {int((mcnemar_table['p_holm'] < alpha).sum())} "
        f"из {len(mcnemar_table)}"
    )
    print(
        f"Различия ROC-AUC меньше MDE = {mde['mde']:.4f} следует считать шумом схемы оценки, "
        f"а не превосходством модели."
    )
    print("Ограничения:")
    print("- Тест открывался один раз на этапе 6; здесь используются только сохранённые предсказания.")
    print(
        "- Гиперпараметры подбирались на train (широкий поиск), поэтому оценки tuned-моделей "
        "оптимистичны относительно честного nested CV."
    )
    print(
        "- Разбиения CV перекрываются (5-fold × 5 повторов): поправка Nadeau & Bengio учитывает это "
        "приближённо, корреляции между фолдами не моделируются."
    )
    print(
        "- Бутстрэп не проверяет калибровку: вероятности используются для ранжирования и порогов "
        "из thresholds.json (пороги подобраны на OOF train)."
    )
    print("- Holm применён внутри семейства гипотез; сравнения между семействами не корректировались.")


def train_frame(cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Train-часть split. Тест не нужен: тестовые числа берутся из артефакта этапа 6."""
    frame = load_split(cfg)[0]
    return frame.drop(columns=[cfg.target]), frame[cfg.target].astype(int)


def run_stats(
    cfg: Config | None = None,
    force: bool = False,
    n_bootstrap: int = N_BOOTSTRAP,
    models: Sequence[TunedModel] | None = None,
    pairs: Sequence[Pair] | None = None,
    payload: dict[str, Any] | None = None,
    train: tuple[pd.DataFrame, pd.Series] | None = None,
) -> dict[str, Path]:
    """Считает все три части этапа 7 и сохраняет таблицы и фигуры.

    ``models``, ``pairs``, ``payload`` и ``train`` можно передать явно (мини-тесты на
    синтетических данных); по умолчанию берутся данные проекта и семь tuned-моделей.
    """
    cfg = cfg or load_config()
    business = load_business_config()
    models = list(models or tuned_models())
    pairs = list(pairs or preregistered_pairs(models))
    if payload is None:
        payload = read_best_params(cfg.tables_dir / BEST_PARAMS_FILE)
    X, y = train if train is not None else train_frame(cfg)
    set_global_seed(cfg.seed)

    print("Этап 7: статистическая значимость различий между tuned-моделями")
    print(f"Train: строк {len(X)}, признаков {X.shape[1]}, доля оттока {float(y.mean()):.4f}")
    print(f"Моделей {len(models)}, зарегистрированных пар {len(pairs)}")
    print(f"alpha = {ALPHA}, бутстрэп-выборок {n_bootstrap}, seed = {cfg.seed}")
    mapping = ", ".join(f"{model.key}→{model.evaluate_key}" for model in models)
    print(f"Ключи: best_params.json → этап 6 (test_predictions.parquet, thresholds.json): {mapping}")
    family_sizes = {family: sum(1 for pair in pairs if pair.family == family) for family in FAMILIES}
    print(f"Пары по семействам: {family_sizes}; поправка Holm внутри семейства и метрики")

    print()
    print("Часть 1: CV на train и попарные corrected resampled t-test")
    cv_frame, ratio_test_train = cv_scores(models, X, y, cfg, payload, force=force)
    print_cv_summary(cv_frame, models, ratio_test_train)
    ttest_table = ttest_frame(cv_frame, pairs, models, ratio_test_train)
    paths = {"ttest": save_table(ttest_table, cfg.tables_dir / TTEST_FILE)}
    paths["cv_scores"] = cfg.tables_dir / CV_SCORES_FILE
    print_ttest_report(ttest_table, ALPHA)
    mde = minimal_detectable_difference(ttest_table, "roc_auc")
    print_mde(mde, ALPHA)

    print()
    print("Часть 2: бутстрэп на тестовых предсказаниях")
    y_test, proba = load_test_predictions(cfg, models)
    thresholds = thresholds_from_json(cfg, models)
    print(f"Тест: строк {y_test.size}, доля оттока {float(y_test.mean()):.4f}; пороги из {THRESHOLDS_FILE}")
    distributions, meta = bootstrap_distributions(
        y_test,
        proba,
        thresholds,
        business,
        [model.key for model in models],
        n_bootstrap,
        cfg.seed,
    )
    ci_table = bootstrap_ci_frame(y_test, proba, thresholds, business, distributions, models)
    diff_table = bootstrap_diff_frame(distributions, pairs, models)
    paths["bootstrap_ci"] = save_table(ci_table, cfg.tables_dir / BOOTSTRAP_CI_FILE)
    paths["bootstrap_diff"] = save_table(diff_table, cfg.tables_dir / BOOTSTRAP_DIFF_FILE)
    print_bootstrap_report(ci_table, diff_table, meta, ALPHA)

    print()
    print("Часть 3: McNemar exact на тесте")
    mcnemar_table = mcnemar_frame(y_test, proba, thresholds, pairs, models)
    paths["mcnemar"] = save_table(mcnemar_table, cfg.tables_dir / MCNEMAR_FILE)
    print_mcnemar_report(mcnemar_table, ALPHA)

    print()
    print("Фигуры")
    paths.update(make_figures(cv_frame, ci_table, ttest_table, models, cfg.figures_dir, meta["n_bootstrap"]))

    print()
    print_conclusion(ttest_table, diff_table, mcnemar_table, mde, ALPHA)
    return paths


def main() -> None:
    configure_utf8_output()
    parser = argparse.ArgumentParser(description="Этап 7: статистическая значимость различий между моделями.")
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"пересчитать CV-оценки ({CV_SCORES_FILE}) даже если файл уже есть",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help=f"число бутстрэп-выборок на тесте (по умолчанию {N_BOOTSTRAP})",
    )
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap должен быть не меньше 100: иначе бутстрэп-p не имеет разрешения")
    paths = run_stats(force=args.force, n_bootstrap=args.bootstrap)
    print()
    print("Артефакты этапа 7:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()












