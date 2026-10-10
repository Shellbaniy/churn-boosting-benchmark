"""Baseline (этап 4): опорные числа на train, без обращения к тесту.

Модели: DummyClassifier(prior), LogisticRegression (вариант A) и xgb/lgbm/catboost
с дефолтными параметрами в вариантах A и B. Фолды те же, что в smoke-прогоне.
Метрики считаются на out-of-fold предсказаниях; порог 0.5 для F1/recall/precision
не настраивается (подбор порога по бизнес-стоимости — только этап 6, на train OOF).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import Config, load_config
from src.data import load_split
from src.pipelines import (
    DUMMY_MODEL,
    MODEL_LABELS,
    MODEL_NAMES,
    available_variants,
    build_pipeline,
    make_folds,
)
from src.utils import configure_utf8_output, ensure_dir, set_global_seed

THRESHOLD = 0.5
BASELINE_MODELS: tuple[str, ...] = (DUMMY_MODEL, *MODEL_NAMES)
METRIC_COLUMNS: tuple[str, ...] = ("roc_auc", "pr_auc", "brier", "f1", "recall", "precision")
ABORT_ROC_AUC = 0.86
DUMMY_AUC_TOLERANCE = 0.05


def evaluate_fold(
    model_name: str,
    variant: str,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    cfg: Config,
) -> dict[str, Any]:
    """Обучение на фолде-train и метрики на фолде-valid; препроцессинг внутри Pipeline."""
    pipe = build_pipeline(model_name, variant, cfg=cfg)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    fit_started = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_sec = time.perf_counter() - fit_started

    infer_started = time.perf_counter()
    proba = pipe.predict_proba(X_valid)[:, 1]
    predict_sec = time.perf_counter() - infer_started

    pred = (proba >= THRESHOLD).astype(int)
    return {
        "model": model_name,
        "variant": variant,
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "churn_rate_valid": round(float(y_valid.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_valid, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_valid, proba)), 4),
        "brier": round(float(brier_score_loss(y_valid, proba)), 4),
        "f1": round(float(f1_score(y_valid, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_valid, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_valid, pred, zero_division=0)), 4),
        "fit_sec": round(fit_sec, 2),
        "predict_sec": round(predict_sec, 4),
    }


def aggregate(folds_df: pd.DataFrame) -> pd.DataFrame:
    """Среднее и std по фолдам для каждой модели и варианта."""
    rows: list[dict[str, Any]] = []
    for (model_name, variant), group in folds_df.groupby(["model", "variant"], sort=False):
        row: dict[str, Any] = {
            "model": model_name,
            "label": MODEL_LABELS[model_name],
            "variant": variant,
            "n_folds": len(group),
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = round(float(group[metric].mean()), 4)
            row[f"{metric}_std"] = round(float(group[metric].std(ddof=1)), 4)
        row["fit_sec_mean"] = round(float(group["fit_sec"].mean()), 2)
        row["predict_sec_mean"] = round(float(group["predict_sec"].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def report(summary: pd.DataFrame, churn_rate: float) -> None:
    """Печатает таблицу по ROC-AUC и короткие выводы по опорным числам."""
    show = summary[
        [
            "model",
            "variant",
            "roc_auc_mean",
            "roc_auc_std",
            "pr_auc_mean",
            "brier_mean",
            "f1_mean",
            "recall_mean",
            "precision_mean",
            "fit_sec_mean",
            "predict_sec_mean",
        ]
    ]
    print("\nBaseline: среднее по фолдам, отсортировано по ROC-AUC:")
    print(show.to_string(index=False))
    print(
        "Порог 0.5 для F1/recall/precision не настраивался; fit_sec и predict_sec — секунды на фолд."
    )

    best_auc = float(summary["roc_auc_mean"].max())
    if best_auc > ABORT_ROC_AUC:
        raise SystemExit(
            f"Остановка: лучший ROC-AUC = {best_auc:.4f} выше {ABORT_ROC_AUC:.2f} — ищи утечку данных."
        )

    dummy_auc = float(summary.loc[summary["model"] == DUMMY_MODEL, "roc_auc_mean"].iloc[0])
    if abs(dummy_auc - 0.5) > DUMMY_AUC_TOLERANCE:
        raise SystemExit(
            f"Остановка: Dummy(prior) ROC-AUC = {dummy_auc:.4f}, ожидалось около 0.5."
        )
    print(f"Dummy(prior): ROC-AUC = {dummy_auc:.4f} — sanity check пройден (около 0.5).")

    logreg = summary.loc[summary["model"] == "logreg"].iloc[0]
    best_boosting = summary.loc[summary["model"].isin([m for m in MODEL_NAMES if m != "logreg"])].iloc[0]
    gap = float(best_boosting["roc_auc_mean"]) - float(logreg["roc_auc_mean"])
    print(
        f"Разрыв LogisticRegression (вариант A) = {float(logreg['roc_auc_mean']):.4f} и лучшего бустинга "
        f"{best_boosting['label']} (вариант {best_boosting['variant']}) = "
        f"{float(best_boosting['roc_auc_mean']):.4f}: {gap:+.4f} ROC-AUC."
    )

    unstable = summary.loc[summary["roc_auc_std"].fillna(0.0).idxmax()]
    print(
        f"Наибольшая дисперсия между фолдами: {unstable['label']} (вариант {unstable['variant']}) — "
        f"std ROC-AUC = {float(unstable['roc_auc_std']):.4f}; у лучшего бустинга std = "
        f"{float(best_boosting['roc_auc_std']):.4f}."
    )
    print(
        f"Порог 0.5 (не настраивался) при доле оттока {churn_rate:.4f}: LogReg recall = "
        f"{float(logreg['recall_mean']):.4f}, precision = {float(logreg['precision_mean']):.4f}; "
        f"{best_boosting['label']} (вариант {best_boosting['variant']}) recall = "
        f"{float(best_boosting['recall_mean']):.4f}, precision = {float(best_boosting['precision_mean']):.4f}. "
        "Отток — меньшинство, поэтому фиксированный порог 0.5 смещает предсказания к мажоритарному "
        "классу: recall систематически ниже precision, часть уходящих клиентов в список удержания "
        "не попадает. Подбор порога по бизнес-стоимости — этап 6, только на train OOF."
    )


def run_baseline(cfg: Config | None = None) -> pd.DataFrame:
    """Дефолтные модели и Dummy на общих фолдах train; тест не читается."""
    configure_utf8_output()
    cfg = cfg or load_config()
    set_global_seed(cfg.seed)

    train = load_split(cfg)[0]
    X = train.drop(columns=[cfg.target])
    y = train[cfg.target]

    print("Baseline: считаем только на train, тестовая выборка не читается.")
    print(f"Train: строк = {len(X)}, признаков = {X.shape[1]}, доля оттока = {y.mean():.4f}")
    folds = make_folds(X, y, cfg)
    print(
        f"Фолды: StratifiedKFold(n_splits={cfg.cv_splits}, shuffle=True, random_state={cfg.seed}) — "
        "те же фолды, что в smoke-прогоне, для всех моделей и вариантов."
    )
    print(
        f"seed = {cfg.seed}, n_jobs = {cfg.n_jobs}; параметры моделей дефолтные; "
        f"порог для F1/recall/precision = {THRESHOLD} (не настраивался)."
    )

    rows: list[dict[str, Any]] = []
    for model_name in BASELINE_MODELS:
        for variant in available_variants(model_name):
            fold_rows: list[dict[str, Any]] = []
            for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
                row = evaluate_fold(model_name, variant, X, y, train_idx, valid_idx, cfg)
                row["fold"] = fold_id
                fold_rows.append(row)
            rows.extend(fold_rows)
            frame = pd.DataFrame(fold_rows)
            print(
                f"{MODEL_LABELS[model_name]:<18} вариант {variant}: ROC-AUC = "
                f"{frame['roc_auc'].mean():.4f} ± {frame['roc_auc'].std(ddof=1):.4f}, "
                f"обучение = {frame['fit_sec'].mean():.2f} с/фолд, "
                f"предсказание = {frame['predict_sec'].mean():.4f} с/фолд"
            )

    folds_df = pd.DataFrame(rows)[
        [
            "fold",
            "model",
            "variant",
            "n_train",
            "n_valid",
            "churn_rate_valid",
            *METRIC_COLUMNS,
            "fit_sec",
            "predict_sec",
        ]
    ]
    summary = aggregate(folds_df)

    ensure_dir(cfg.tables_dir)
    folds_out = cfg.tables_dir / "baseline_folds.csv"
    summary_out = cfg.tables_dir / "baseline.csv"
    folds_df.to_csv(folds_out, index=False)
    summary.to_csv(summary_out, index=False)
    print(f"\nМетрики по фолдам сохранены: {folds_out}")
    print(f"Среднее и std по фолдам сохранены: {summary_out}")

    report(summary, float(y.mean()))
    return summary


def main() -> None:
    run_baseline()


if __name__ == "__main__":
    main()

