"""Общие утилиты: seeds, логирование, ввод/вывод."""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


def configure_utf8_output() -> None:
    """UTF-8 для stdout/stderr: при редиректе в файл Windows берёт cp1252."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        ensure_dir(log_file.parent)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
