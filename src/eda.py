"""EDA на train: обязательные фигуры и таблицы. Тестовые строки не используются."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import Config, load_config
from src.data import load_clean, load_raw
from src.utils import configure_utf8_output, ensure_dir

CLASS_LABELS = {0: "Остались (0)", 1: "Ушли (1)"}

plt.rcParams["axes.unicode_minus"] = False


def _train_indices(cfg: Config) -> np.ndarray:
    path = cfg.split_indices
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла индексов split: {path}. Сначала выполните `make data`."
        )
    return np.load(path)["train_idx"]


def load_train(cfg: Config) -> pd.DataFrame:
    """Очищенный train (target 0/1, без customerID)."""
    df = load_clean(cfg)
    return df.iloc[_train_indices(cfg)].copy()


def load_train_raw(cfg: Config) -> pd.DataFrame:
    """Сырой train: нужен, чтобы видеть исходные пропуски TotalCharges."""
    raw = load_raw(cfg)
    return raw.iloc[_train_indices(cfg)].copy()


def class_balance(train: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    counts = train[cfg.target].value_counts().sort_index()
    shares = counts / counts.sum()
    return pd.DataFrame(
        {
            "class": [CLASS_LABELS[int(i)] for i in counts.index],
            "n": counts.to_numpy(),
            "share": shares.round(4).to_numpy(),
        }
    )


def missing_totalcharges_table(train_raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    tc = pd.to_numeric(train_raw["TotalCharges"], errors="coerce")
    miss = train_raw.loc[tc.isna()].copy()
    miss["TotalCharges_raw"] = train_raw.loc[tc.isna(), "TotalCharges"]
    wanted = [
        cfg.id_column,
        "tenure",
        "MonthlyCharges",
        "TotalCharges_raw",
        "Contract",
        "InternetService",
        "PaymentMethod",
    ]
    cols = [c for c in wanted if c in miss.columns]
    return miss[cols].reset_index(drop=True)


def churn_by_category(train: pd.DataFrame, cfg: Config, column: str) -> pd.DataFrame:
    grouped = train.groupby(column, observed=True)[cfg.target].agg(n="size", churn_rate="mean")
    grouped["churn_rate"] = grouped["churn_rate"].round(4)
    return grouped.reset_index().rename(columns={column: "category"})


def save_tables(
    balance: pd.DataFrame,
    miss_table: pd.DataFrame,
    cat_tables: dict[str, pd.DataFrame],
    cfg: Config,
) -> list[Path]:
    ensure_dir(cfg.tables_dir)
    paths: list[Path] = []

    p_balance = cfg.tables_dir / "eda_class_balance.csv"
    balance.to_csv(p_balance, index=False)
    paths.append(p_balance)

    p_miss = cfg.tables_dir / "eda_missing_totalcharges.csv"
    miss_table.to_csv(p_miss, index=False)
    paths.append(p_miss)

    combined = pd.concat(
        [table.assign(feature=name) for name, table in cat_tables.items()], ignore_index=True
    )
    p_cat = cfg.tables_dir / "eda_churn_by_category.csv"
    combined[["feature", "category", "n", "churn_rate"]].to_csv(p_cat, index=False)
    paths.append(p_cat)
    return paths


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_imbalance(balance: pd.DataFrame, cfg: Config) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(balance["class"], balance["n"], color=["#4c72b0", "#c44e52"])
    for bar, share, n in zip(bars, balance["share"], balance["n"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{n} ({share:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_xlabel("Класс (Churn)")
    ax.set_ylabel("Число клиентов")
    ax.set_title("Дисбаланс классов (train)")
    ax.margins(y=0.15)
    return _save_figure(fig, cfg.figures_dir / "eda_class_imbalance.png")


def plot_missing_totalcharges(train_raw: pd.DataFrame, cfg: Config) -> Path:
    tc = pd.to_numeric(train_raw["TotalCharges"], errors="coerce")
    missing = tc.isna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    counts = train_raw.loc[missing, "tenure"].value_counts().sort_index()
    if len(counts) == 0:
        axes[0].text(0.5, 0.5, "В train пропусков TotalCharges нет", ha="center", va="center")
    else:
        axes[0].bar([str(int(v)) for v in counts.index], counts.to_numpy(), color="#c44e52")
        for i, v in enumerate(counts.to_numpy()):
            axes[0].text(i, v, str(int(v)), ha="center", va="bottom")
    axes[0].set_xlabel("tenure (месяцев)")
    axes[0].set_ylabel("Число пропусков")
    axes[0].set_title("Пропуски TotalCharges по tenure")

    present = ~missing
    axes[1].scatter(
        train_raw.loc[present, "tenure"], tc[present], s=6, alpha=0.25,
        color="#4c72b0", label="TotalCharges известен",
    )
    axes[1].scatter(
        train_raw.loc[missing, "tenure"], np.zeros(int(missing.sum())),
        s=60, marker="x", color="#c44e52", label="пропуск (значение отсутствует)",
    )
    axes[1].set_xlabel("tenure (месяцев)")
    axes[1].set_ylabel("TotalCharges, исходное значение")
    axes[1].set_title("TotalCharges vs tenure")
    axes[1].legend(loc="upper left")
    axes[1].text(
        0.02, 0.72, "маркеры 'x' стоят на y = 0\nтолько как отметка пропуска",
        transform=axes[1].transAxes, fontsize=8, va="top",
    )
    fig.tight_layout()
    return _save_figure(fig, cfg.figures_dir / "eda_missing_totalcharges.png")


def plot_churn_rate(
    table: pd.DataFrame, cfg: Config, feature: str, xlabel: str, filename: str
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(table["category"].astype(str), table["churn_rate"] * 100, color="#4c72b0")
    for bar, rate, n in zip(bars, table["churn_rate"], table["n"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{rate:.1%}\n(n={int(n)})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Доля оттока, %")
    ax.set_title(f"Доля оттока по признаку {feature} (train)")
    ax.margins(y=0.2)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _save_figure(fig, cfg.figures_dir / filename)


def plot_tenure_by_class(train: pd.DataFrame, cfg: Config) -> Path:
    data = train.assign(Класс=train[cfg.target].map(CLASS_LABELS))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(
        data=data, x="tenure", hue="Класс", stat="density", common_norm=False,
        bins=30, element="step", fill=True, ax=ax,
    )
    ax.set_xlabel("tenure (месяцев)")
    ax.set_ylabel("Плотность распределения")
    ax.set_title("Распределение tenure по классам (train)")
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Класс")
    return _save_figure(fig, cfg.figures_dir / "eda_tenure_by_class.png")


def print_conclusions(
    balance: pd.DataFrame,
    train_raw: pd.DataFrame,
    cat_tables: dict[str, pd.DataFrame],
    cfg: Config,
) -> None:
    n_total = int(balance["n"].sum())
    n_churn = int(balance.loc[balance["class"] == CLASS_LABELS[1], "n"].iloc[0])
    churn_rate = n_churn / n_total
    print(
        f"Train: клиентов = {n_total}, ушло = {n_churn} ({churn_rate:.2%}), "
        f"осталось = {n_total - n_churn} ({1 - churn_rate:.2%})"
    )
    print(
        f"Accuracy константного прогноза 'все остаются' = {1 - churn_rate:.4f} "
        f"({1 - churn_rate:.2%}); recall такого прогноза = 0."
    )
    print(
        "Вывод: accuracy не подходит. При доле оттока около 26% прогноз 'все остаются' даёт "
        f"~{1 - churn_rate:.0%} accuracy, но не находит ни одного ушедшего клиента. "
        "Основная метрика — ROC-AUC, а бизнес-решение строится по порогу."
    )

    tc = pd.to_numeric(train_raw["TotalCharges"], errors="coerce")
    n_miss = int(tc.isna().sum())
    n_miss_tenure0 = int((tc.isna() & (train_raw["tenure"] == 0)).sum())
    median_present = float(tc.median())
    n_miss_total = int(pd.to_numeric(load_raw(cfg)["TotalCharges"], errors="coerce").isna().sum())
    print(
        f"\nTotalCharges: пропусков всего = {n_miss_total}, в train = {n_miss} "
        f"(остальные попали в тест и не используются); в train все {n_miss_tenure0} "
        f"приходятся на tenure == 0; медиана по непустым = {median_present:.2f}."
    )
    print(
        "Обоснование заполнения нулём: все пропуски приходятся на tenure == 0 — это новые "
        "клиенты, которые ещё не платили, поэтому ожидаемые накопленные платежи равны 0 "
        f"(TotalCharges ~ tenure x MonthlyCharges). Медиана ({median_present:.2f}) описывает "
        "клиента в середине срока и завысила бы признак. Пропусков всего "
        f"{n_miss} из {len(train_raw)} ({n_miss / len(train_raw):.2%}) — влияние мало."
    )

    for feature, table in cat_tables.items():
        print(f"\nДоля оттока по {feature}:")
        for _, row in table.iterrows():
            print(f"  {row['category']}: n = {int(row['n'])}, доля оттока = {row['churn_rate']:.2%}")


def main() -> None:
    configure_utf8_output()
    cfg = load_config()
    ensure_dir(cfg.figures_dir)
    ensure_dir(cfg.tables_dir)

    train = load_train(cfg)
    train_raw = load_train_raw(cfg)
    total_rows = len(load_clean(cfg))
    print(
        f"EDA только на train: {len(train)} строк из {total_rows}; "
        "тестовые строки не используются."
    )

    balance = class_balance(train, cfg)
    miss_table = missing_totalcharges_table(train_raw, cfg)
    cat_tables = {
        column: churn_by_category(train, cfg, column)
        for column in ("Contract", "InternetService", "PaymentMethod")
    }

    table_paths = save_tables(balance, miss_table, cat_tables, cfg)
    print(f"Сохранено таблиц: {len(table_paths)}")
    for path in table_paths:
        print(f"  {path.name}")

    figure_paths = [
        plot_class_imbalance(balance, cfg),
        plot_missing_totalcharges(train_raw, cfg),
        plot_churn_rate(
            cat_tables["Contract"], cfg, "Contract", "тип контракта", "eda_contract_churn.png"
        ),
        plot_tenure_by_class(train, cfg),
        plot_churn_rate(
            cat_tables["InternetService"], cfg, "InternetService",
            "тип интернет-услуги", "eda_internetservice_churn.png",
        ),
        plot_churn_rate(
            cat_tables["PaymentMethod"], cfg, "PaymentMethod",
            "способ оплаты", "eda_paymentmethod_churn.png",
        ),
    ]
    print(f"Сохранено фигур: {len(figure_paths)}")
    for path in figure_paths:
        print(f"  {path.name}")

    print()
    print_conclusions(balance, train_raw, cat_tables, cfg)


if __name__ == "__main__":
    main()
