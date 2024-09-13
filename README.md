README.md

#### Проект. Рекомендательные системы в банковской сфере
 - Цель: Предсказать, какой банковский продукт предложить клиенту.
 - Основные задачи: Анализ данных о клиентах, определение важных метрик, моделирование, продуктивизация модели, настройка мониторинга и дообучения.

#### Структура проекта

| # | Задача | Решение | Источник |
|:--:|:---------|:-----------|:------------|
| 1 | Исследование данных | Проведен первичный анализ данных в Jupyter Notebook, приведены описание действий и выводы| [Jupyter NB: eda.ipynb](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/eda.ipynb)  
| 2 | Подготовка инфраструктуры | Создан скрипт запуска MLflow, впоследствии включен в Docker compose с остальными сервисами|	[Запуск: run_mlflow.sh](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/run_mlflow.sh)
|3| Трансляция | Описание метрик приводится в Jupyter Notebook вместе с исследовательским анализом и обоснованием выбора метрик |[Jupyter NB: eda.ipynb](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/eda.ipynb)  
|4| Моделирование | Проведен эксперимент с логированием в MLFlow, подготовлен sklearn пайплайн и обученная модель. | [Jupyter NB experiments.ipynb](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/experiments.ipynb), модель: fitted_model.plk
|5| Продуктивизация | Модель обернута в сервис FastAPI, с реализацией ответов на внешние запросы, все реализовано в Docker compose|	[Dockerfile](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/Dockerfile), [docker-compose.yaml](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/docker-compose.yaml), [сервис: app.py](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/app.py)
|6| Мониторинг | Реализована push-модель контроль сервиса на основе сервера graphite, передающего метрики в grafana для мониторинга | [README.md](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/README.md)
|7| Документация | Процесс обработки данных, создания модели, её выкатки и сопровождения описаны ниже | [README.md](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/README.md)
|8|Требования и среда | Проект проводился в виртуальном окружении, зависимости зафиксированы, воспроизводимость экспериментов соблюдалась | [requirements.txt](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/requirements.txt)


