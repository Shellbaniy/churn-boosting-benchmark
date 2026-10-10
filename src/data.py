"""Загрузка, очистка и стратифицированный split датасета Telco Customer Churn."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Config, load_config
from src.utils import configure_utf8_output, ensure_dir


def load_raw(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.raw_data)


def clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing = df["TotalCharges"].isna()
    if missing.any():
        n_bad = int((missing & (df["tenure"] != 0)).sum())
        assert n_bad == 0, (
            "Пропуски TotalCharges найдены не только при tenure == 0 "
            f"({n_bad} строк). Остановка: заполнение нулём не выполнено."
        )
        df.loc[missing, "TotalCharges"] = 0.0

    if cfg.id_column in df.columns:
        df = df.drop(columns=[cfg.id_column])

    df[cfg.target] = (df[cfg.target] == "Yes").astype(int)
    return df


def split_indices(df: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=cfg.test_size,
        stratify=df[cfg.target].to_numpy(),
        random_state=cfg.seed,
    )
    return train_idx, test_idx


def save_split_indices(train_idx: np.ndarray, test_idx: np.ndarray, cfg: Config) -> Path:
    ensure_dir(cfg.processed_dir)
    np.savez(cfg.split_indices, train_idx=train_idx, test_idx=test_idx)
    return cfg.split_indices


def load_clean(cfg: Config) -> pd.DataFrame:
    return clean(load_raw(cfg), cfg)


def load_split(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test по сохранённым индексам. Test трогаем только на этапе 6."""
    df = load_clean(cfg)
    indices = np.load(cfg.split_indices)
    train = df.iloc[indices["train_idx"]].copy()
    test = df.iloc[indices["test_idx"]].copy()
    return train, test


def main() -> None:
    configure_utf8_output()
    cfg = load_config()
    raw = load_raw(cfg)
    print(f"Загружен датасет: строк = {len(raw)}, колонок = {raw.shape[1]}")

    tc = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    n_blank = int(raw["TotalCharges"].astype(str).str.strip().eq("").sum())
    n_missing = int(tc.isna().sum())
    n_missing_tenure0 = int((tc.isna() & (raw["tenure"] == 0)).sum())
    print(
        "Пропуски TotalCharges: пустых строк = "
        f"{n_blank}, NaN после to_numeric = {n_missing}, "
        f"из них при tenure == 0 = {n_missing_tenure0}"
    )
    assert n_missing == n_missing_tenure0, (
        "Пропуски TotalCharges есть не только при tenure == 0 "
        f"({n_missing - n_missing_tenure0} строк). Остановка, заполнение не выполнено."
    )

    dropped = [cfg.id_column] if cfg.id_column in raw.columns else []
    print(f"Удалённые колонки: {dropped if dropped else 'нет'}")

    df = clean(raw, cfg)
    print("Заполнение TotalCharges нулём выполнено: все пропуски приходятся на tenure == 0.")

    n_dup = int(df.duplicated().sum())
    print(f"Полных дубликатов строк (после удаления customerID): {n_dup} (не удаляются).")

    train_idx, test_idx = split_indices(df, cfg)
    save_split_indices(train_idx, test_idx, cfg)

    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    print(f"Всего после очистки: строк = {len(df)}, колонок = {df.shape[1]}")
    print(
        f"Train: строк = {len(train)}, колонок = {train.shape[1]}, "
        f"доля оттока = {train[cfg.target].mean():.4f}"
    )
    print(
        f"Test:  строк = {len(test)}, колонок = {test.shape[1]}, "
        f"доля оттока = {test[cfg.target].mean():.4f}"
    )
    print(f"Индексы split сохранены: {cfg.split_indices}")

    ensure_dir(cfg.tables_dir)
    summary = pd.DataFrame(
        [
            {
                "n_rows": len(df),
                "n_cols": df.shape[1],
                "n_features": df.shape[1] - 1,
                "churn_rate": round(float(df[cfg.target].mean()), 4),
                "missing_total_charges_raw": n_missing,
                "missing_total_charges_tenure0": n_missing_tenure0,
                "duplicate_rows": n_dup,
                "dropped_columns": ";".join(dropped),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_churn_rate": round(float(train[cfg.target].mean()), 4),
                "test_churn_rate": round(float(test[cfg.target].mean()), 4),
            }
        ]
    )
    out_path = cfg.tables_dir / "data_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Сводка данных сохранена: {out_path}")


if __name__ == "__main__":
    main()
