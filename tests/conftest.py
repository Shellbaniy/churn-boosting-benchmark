"""Общие фикстуры тестов. Тестовая часть split не используется нигде, кроме этапа 6."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, load_config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.utils import configure_utf8_output  # noqa: E402

# При редиректе вывода pytest в файл Windows отдаёт cp1252: русские print падали бы с UnicodeEncodeError.
configure_utf8_output()


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def train_frame(cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    train = load_split(cfg)[0]
    return train.drop(columns=[cfg.target]), train[cfg.target]


@pytest.fixture(scope="session")
def first_fold(train_frame, cfg: Config) -> tuple:
    X, y = train_frame
    cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.seed)
    return next(iter(cv.split(X, y)))


@pytest.fixture(scope="session")
def small_train(train_frame, cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Компактная подвыборка train для быстрых тестов детерминизма."""
    X, y = train_frame
    sample = X.sample(n=300, random_state=cfg.seed)
    return sample, y.loc[sample.index]

