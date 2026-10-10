# churn-boosting-benchmark

Сравнительный анализ алгоритмов градиентного бустинга XGBoost, LightGBM и CatBoost
на задаче прогнозирования оттока клиентов (Telco Customer Churn).

## Идея

На датасете такого размера (7043 строки) разница между boosting-алгоритмами, скорее
всего, находится в пределах статистического шума. Вывод строится на статистических
тестах, а не на разнице метрик в третьем знаке.

## Структура репозитория

```
configs/     конфигурации (данные, CV, поиск, бизнес-допущения)
src/         модули пайплайна
tests/       pytest-тесты
notebooks/   EDA и финальные графики
reports/     figures, tables, logs, mlflow
article/     текст статьи
data/raw/    исходный CSV
```

## Воспроизведение

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Windows: вместо `make` используйте `make.bat`. На Linux/macOS — `make`.

```bash
make data        # загрузка, очистка, split
make eda         # EDA и графики
make baseline    # baseline-модели
make tune-smoke  # прогон Optuna на 5 trials
make tune        # полный прогон Optuna
make benchmark   # финальный бенчмарк на тесте
make stats       # статистическая значимость
make explain     # SHAP
make test        # pytest
```

## Результаты

Заполняется после финального бенчмарка. Все числа берутся из `reports/tables/`.

## Ограничения

- Гиперпараметры подбирались на тех же данных, что и оценка; итоговая метрика слегка
  оптимистична. Nested CV рассматривается как возможное улучшение.
- Бизнес-метрика опирается на сценарные допущения о стоимости, а не на фактические данные.
