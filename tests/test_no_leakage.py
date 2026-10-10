"""Проверки отсутствия утечки: препроцессор учится только на фолде-train.

Обученные параметры препроцессора (mean/scale у StandardScaler, список категорий у
OneHotEncoder и CategoricalEncoder) сравниваются со статистикой фолда-train и явно не
совпадают со статистикой полного train. У каждой проверки есть контроль: если бы
валидационные строки участвовали в fit, тест это заметил бы.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.pipelines import (
    UNKNOWN_TOKEN,
    CategoricalEncoder,
    available_variants,
    build_pipeline,
    categorical_columns,
    numeric_columns,
)


def _fold_frames(X: pd.DataFrame, y: pd.Series, fold: tuple) -> tuple:
    train_idx, valid_idx = fold
    return X.iloc[train_idx], y.iloc[train_idx], X.iloc[valid_idx], y.iloc[valid_idx]


def _scaler(pipe):
    return pipe.named_steps["preprocess"].named_transformers_["num"]


def _ohe(pipe):
    return pipe.named_steps["preprocess"].named_transformers_["cat"]


def _unfitted_step(preprocessor: ColumnTransformer, name: str):
    """Доступ к шагу ColumnTransformer до fit (named_transformers_ появляется только после fit)."""
    return {step_name: step for step_name, step, _ in preprocessor.transformers}[name]


def test_scaler_stats_computed_on_fold_train_only(train_frame, first_fold, cfg) -> None:
    X, y = train_frame
    X_train, y_train, _, _ = _fold_frames(X, y, first_fold)
    numeric = numeric_columns(X_train)

    pipe = build_pipeline("logreg", "A", cfg=cfg).fit(X_train, y_train)
    scaler = _scaler(pipe)

    fold_mean = X_train[numeric].mean().to_numpy()
    fold_scale = X_train[numeric].std(ddof=0).to_numpy()
    full_mean = X[numeric].mean().to_numpy()
    full_scale = X[numeric].std(ddof=0).to_numpy()

    print(f"Фолд-train = {len(X_train)} строк, полный train = {len(X)} строк")
    print(f"mean фолда:   {np.round(fold_mean, 4)}")
    print(f"scaler.mean_: {np.round(scaler.mean_, 4)}")
    print(f"mean полного train: {np.round(full_mean, 4)}")

    assert np.allclose(scaler.mean_, fold_mean)
    assert np.allclose(scaler.scale_, fold_scale)
    assert not np.allclose(scaler.mean_, full_mean, atol=1e-6), "scaler.mean_ совпал с полным train"
    assert not np.allclose(scaler.scale_, full_scale, atol=1e-6), "scaler.scale_ совпал с полным train"


def test_ohe_categories_equal_fold_train_values(train_frame, first_fold, cfg) -> None:
    X, y = train_frame
    X_train, y_train, _, _ = _fold_frames(X, y, first_fold)
    categorical = categorical_columns(X_train)

    pipe = build_pipeline("xgb", "A", cfg=cfg).fit(X_train, y_train)
    ohe = _ohe(pipe)

    for index, column in enumerate(categorical):
        fold_values = set(X_train[column].unique())
        print(f"{column}: в фолде = {len(fold_values)}, в препроцессоре = {len(ohe.categories_[index])}")
        assert set(ohe.categories_[index]) == fold_values

    learned = {value for categories in ohe.categories_ for value in categories}
    fold_union = {value for column in categorical for value in X_train[column].unique()}
    assert learned == fold_union, "в OHE попали категории не из фолда-train"


def test_ohe_category_names_come_from_its_own_fit_data(train_frame, first_fold, cfg) -> None:
    X, y = train_frame
    X_train, y_train, _, _ = _fold_frames(X, y, first_fold)
    column, original, mutated = "Contract", "Month-to-month", "Month-to-month__renamed"

    X_mutated = X_train.copy()
    X_mutated[column] = X_mutated[column].where(X_mutated[column] != original, mutated)

    ohe = _ohe(build_pipeline("xgb", "A", cfg=cfg).fit(X_mutated, y_train))
    index = categorical_columns(X_mutated).index(column)
    learned = set(ohe.categories_[index])

    print(f"В фолде-train '{original}' переименован в '{mutated}': OHE выучил {sorted(learned)}")
    print(f"'{original}' присутствует в полном train: {original in set(X[column])}")

    assert mutated in learned
    assert original not in learned, "OHE выучил категорию не из своей обучающей выборки"
    assert original in set(X[column]), "контроль не имеет смысла: значения нет в полном train"


def test_validation_rows_do_not_affect_fitted_preprocessor(train_frame, first_fold, cfg) -> None:
    X, y = train_frame
    X_train, y_train, X_valid, y_valid = _fold_frames(X, y, first_fold)
    numeric = numeric_columns(X_train)
    categorical = categorical_columns(X_train)

    reference = build_pipeline("logreg", "A", cfg=cfg).fit(X_train, y_train)
    ref_mean = _scaler(reference).mean_.copy()
    ref_scale = _scaler(reference).scale_.copy()
    ref_categories = [list(values) for values in _ohe(reference).categories_]

    mutated = X_valid.copy()
    mutated[numeric] = mutated[numeric] * 1000.0 + 777.0
    mutated[categorical] = "UNSEEN_VALUE"

    transformed = reference.named_steps["preprocess"].transform(mutated)
    cat_block = transformed[:, len(numeric):]
    print(f"Изменено валидационных строк: {len(mutated)}; сумма OHE-блока = {abs(cat_block).sum():.6f}")
    assert np.allclose(cat_block, 0.0), "неизвестные категории не игнорируются (handle_unknown)"

    again = build_pipeline("logreg", "A", cfg=cfg).fit(X_train, y_train)
    assert np.array_equal(_scaler(again).mean_, ref_mean)
    assert np.array_equal(_scaler(again).scale_, ref_scale)
    assert [list(values) for values in _ohe(again).categories_] == ref_categories

    leaky = build_pipeline("logreg", "A", cfg=cfg).fit(
        pd.concat([X_train, mutated]), pd.concat([y_train, y_valid])
    )
    leaky_mean = _scaler(leaky).mean_
    delta = float(np.abs(leaky_mean - ref_mean).max())
    print(f"Контроль: при включении валидации в fit mean сдвигается, Δmax = {delta:.4f}")
    assert not np.allclose(leaky_mean, ref_mean, atol=1e-6), "контроль пустой: утечка не была бы замечена"


def test_native_encoder_learns_categories_from_fold_and_hides_unknown(train_frame, first_fold) -> None:
    X, y = train_frame
    X_train, _, _, _ = _fold_frames(X, y, first_fold)
    column = categorical_columns(X_train)[0]

    encoder = CategoricalEncoder(mode="category").fit(X_train)
    for name, learned in encoder.categories_.items():
        assert set(learned) == set(X_train[name].unique())

    probe = X_train.iloc[:5].copy()
    probe[column] = "UNSEEN_VALUE"
    as_category = encoder.transform(probe)[column]
    as_string = CategoricalEncoder(mode="string").fit(X_train).transform(probe)[column]

    print(f"Режим category: неизвестное -> NaN = {int(as_category.isna().sum())} из {len(as_category)}")
    print(f"Режим string: неизвестное -> {as_string.iloc[0]!r}, dtype = {as_string.dtype}")

    assert as_category.isna().all()
    assert (as_string == UNKNOWN_TOKEN).all()
    assert as_string.isna().sum() == 0, "CatBoost отклоняет NaN в cat_features"


def test_preprocessing_lives_inside_pipeline(cfg) -> None:
    for model_name in ("logreg", "xgb", "lgbm", "catboost"):
        for variant in available_variants(model_name):
            pipe = build_pipeline(model_name, variant, cfg=cfg)
            names = [name for name, _ in pipe.steps]
            print(f"{model_name}/{variant}: шаги = {names}")
            assert len(names) == 2, f"{model_name}/{variant}: препроцессинг вне Pipeline ({names})"
            assert names[0] == ("preprocess" if variant == "A" else "categorical")
            assert names[-1] == "model"


def test_scaling_only_for_logistic_regression(cfg) -> None:
    for model_name in ("xgb", "lgbm", "catboost"):
        preprocessor = build_pipeline(model_name, "A", cfg=cfg).named_steps["preprocess"]
        numeric_step = _unfitted_step(preprocessor, "num")
        print(f"{model_name}/A: числовой шаг = {numeric_step!r}")
        assert numeric_step == "passthrough", f"{model_name}/A: числовые признаки масштабируются"
    scaler = _unfitted_step(build_pipeline("logreg", "A", cfg=cfg).named_steps["preprocess"], "num")
    print(f"logreg/A: числовой шаг = {type(scaler).__name__}")
    assert isinstance(scaler, StandardScaler)

