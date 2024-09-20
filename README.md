README.md

#### Проект. Рекомендательные системы в банковской сфере
 - Цель: Предсказать, какой банковский продукт предложить клиенту.
 - Основные задачи: Анализ данных о клиентах, определение важных метрик, моделирование, продуктивизация модели, настройка мониторинга и дообучения.  

#### Структура проекта

| # | Задача | Решение | Источник |
|:--:|:---------|:-----------|:------------|
| 1 | Исследование данных | Проведен первичный анализ данных в Jupyter Notebook, приведены описание действий и выводы| [Jupyter NB: eda.ipynb](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/eda.ipynb)  
| 2 | Подготовка инфраструктуры | Создан скрипт запуска MLflow, включен в Docker compose с остальными сервисами| Запуск: [docker-compose.yaml](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/docker-compose.yaml), [run_mlflow.sh](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/run_mlflow.sh)
|3| Трансляция | Описание метрик приводится в Jupyter Notebook вместе с исследовательским анализом и обоснованием выбора метрик |[Jupyter NB: eda.ipynb](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/eda.ipynb)  
|4| Моделирование | Проведен эксперимент с логированием в MLFlow, подготовлен sklearn пайплайн и обученная модель. Сервис пайплайна обработки данных и обучения модели реализован на платформе Airflow| [Jupyter NB experiments.ipynb](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/experiments.ipynb), модель: fitted_model.plk, [Airflow DAG: pipeline.py](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/airflow/dags/pipeline.py)
|5| Продуктивизация | Модель обернута в сервис FastAPI, с реализацией ответов на внешние запросы, все реализовано в Docker compose|	[docker-compose.yaml](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/docker-compose.yaml), [сервис: app.py](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/app.py)
|6| Мониторинг | Реализована push-модель контроль сервиса на основе сервера graphite, передающего метрики в grafana для мониторинга | [README.md](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/README.md)
|7| Документация | Процесс обработки данных, создания модели, её выкатки и сопровождения описаны ниже | [README.md](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/README.md)
|8|Требования и среда | Проект проводился в виртуальном окружении, зависимости зафиксированы, воспроизводимость экспериментов соблюдалась | [requirements.txt](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/requirements.txt)

#### Рабочий процесс
Датасет представляет собой данные испанского банка о приобретениях продуктов банка пользователями с '2015-01-28' по '2016-05-28' (17 периодов). Информация о продуктах представлена бинарными переменными, их всего 24, каждая из которых отражает пользование данным продуктом в текущем периоде.   
  
Целью проекта является выработка рекомендаций. Базовым решением является предсказание бинарных таргетов (купит / не купит) для каждого продутка, т.е. решением задачи многозначной бинарной классификации. Тогда рекомендациями для пользователя будут как набор продуктов, которые пользователь купит в следующем периоде, так и набор вероятностей, с которой пользователь приобретет эти продукты.  
  
Описание этапа EDA приведено в тетради [eda.ipynb](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/eda.ipynb).

Данные необработанные, не соответствуют типам, заметное количество пропущенных значений.    
Было принято решение создать пайплайн с предобработкой "сырых" данных, на выходе которого будут бинарные предсказания.  
Датасет большой, переменных много: 3 числовых, 17 категориальных, 24 бинарных категориальных, соответствующих купленным продуктам. Таргет генерируется как те же 24 продукта, купленные или не купленные в следующем периоде.   
В данных наблюдается сильный дисбаланс: большое количество "пустых" наблюдений, малое количество покупок (0.18%), неравномерно распределенное по продуктам.  
В условиях слабого представительства целевого класса в таргете выбраны Recall, Precision, F1, ROC-AUC, с акцентом на микро-усреднение метрик (отражение общей картины без предпочтения каких-либо продуктов).  
В Jupyter проводится анализ отдельных переменных, вывод закономерностей, отмечается слабая зависимость целевых переменных от признаков.  
Препроцессинг данных включает стандартные шаги - удаление пропущенных значений, кодирование категориальных переменных, устранение асимметрии распределения ключевой переменной (income),нормирование числовых признаков.  
Препроцессинг завернут в sklearn пайплайн.

Описание этапа моделирования приведено в тетради [experiments.ipynb](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/experiments.ipynb).  
Данные были подготовлены для большинства моделей, выбран "коробочный" Catboost Classifier c "минимальными" параметрами для экономии времени, который был использован в качестве базовой модели для MultiOutputClassifier библиотеки sklearn.
Шаг обучения модели был присоединен к общему пайплайну.
Для логирования используется сервер MLFlow. На стадии моделирования в Jupyter (experiments.ipynb) сервис можно запустить отдельно скриптом run_mlflow.sh. В тетради инициализируется эксперимент, подготавливаются данные и обучается модель. Параметры, модель, образцы данных, метрики - все логируется в MLFlow.  
Полученные метрики сравниваются с константными предсказаниями.  

Инфраструктура сервиса представлена в едином контейнере Docker ([docker-compose.yaml](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/docker-compose.yaml)):  
 - [FastAPI: app.py](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/app.py) - веб-сервис предсказаний,
 - MLFlow: логирование экспериментов,  
 - Graphit: сбор метрик из кода веб-сервиса,  
 - Grafana: мониторинг работы сервиса,  
 - Airflow: пайплайн: вход: загрузка "сырых" данны, выход: обученная модель.  

Сервис предсказаний реализован с помощью фреймворка FastAPI.  
Интерфейс включает 3 эндпоинта:
 - "/": для проверки статуса сервиса
 - "/predict": непосредственно для предсказаний, принимает POST-запрос с необработанными json-данными, выдает предсказания на руском языке в виде словаря: {"Название продукта": [0,1]}
 - "/random": для тестирования сервиса в диапазоне случайных значений, выдает случайное предсказание
 - "/predict_proba": выдает рекомендации на основе вероятностей, принимает POST-запрос с необработанными json-данными, выдает рекомендации на руском языке в виде словаря: {"Название продукта": <вероятность от установленного порога (0.01) до 1>}

Сбор метрик производится в python коде проекта и передается клиенту сервиса мониторинга Graphite.  
В проекте производится генерация следующих метрик:
 - 'bank-rs.system.cpu_load': текущая нагрузка на CPU
 - 'bank-rs.system.memory_free_mb': свободная память
 - 'bank-rs.system.up_time': время работы сервиса
 - 'bank-rs.requests' - общее количество запросов
 - 'response_code.200': счетчик успешных HTTP-запросов
 - 'response_time': время отклика (время получения предсказания)
 - 'target.*': непосредственно полученные предсказания (счетчик положительных предсказаний для каждого таргета)
  
Средства визуализации Graphite довольно скудны, поэтому для отображения используется платформа Grafana.  
Все перечисленные метрики выводятся в виде настраиваемых, масштабируемых, pretty-looking элементов визуализации:  счетчиков продуктов в форме гистограммы, time-series графиков и отображения технических метрик.
Для имитации нагрузки на сервис используется скрипт run_mimic_load, выдавая случайные предсказания.  
[Скриншот Grafana ](https://github.com/vvbelyanin/mle-sprint6-project/blob/main/grafana/Grafana_screenshot.jpg)

#### Основной итог
В проекте удалось реализовать сервис предсказаний для рекомендации клиентам банковских продуктов.  Был построен пайплайн обрабоки данных и обучения модели, развернута структура логирования и проведения экспериментов. Основной и вспомогательные сервис поднимаются в едином контейнере docker, их работа контролируется метриками с визуализацией.
