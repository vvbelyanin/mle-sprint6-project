Instructions.md

**Проект. Рекомендательные системы в банковской сфере**  
**Технические инструкции по выполнению проекта**    

Среда:  
Ubuntu 22.04.3 LTS  
Python 3.10.12  
VS Code: 1.93.0

В командной строке:    
```
# обновление установленных пакетов
sudo apt-get update    

python3 -m pip install --upgrade pip

# установка пакета виртуального окружения Python
sudo apt-get install python3.10-venv    

# создание виртуальной среды
python3.10 -m venv .venv_sprint_6  

# активирование окружения 
source .venv_sprint_6/bin/activate    

# каталог для временных файлов
mkdir -p tmp

# фиксация конкретных версий пакетов
pip install -r requirements.txt    

# в текущей папке должен быть файл .env со следующими переменными
DB_DESTINATION_*=
S3_*=
AWS_*=
MLFLOW_S3_ENDPOINT_URL=
TRACKING_SERVER_CONN=
*_PORT, *_USER, *_PASS etc

# порты по умолчанию:
# 80 -Graphite
# 3000 - Grafana
# 5000 - MLflow
# 8000 - FastAPI
# 8080 - Airflow
# 8793 - Airflow worker_log_server_port
# 8794 - Airflow trigger_log_server_port

# Загрузка данных вручную
# при необходимости установить unzip
sudo apt install unzip

# создать каталог и перейти туда
mkdir data && cd data

# скачать файл
curl -L $(yadisk-direct https://disk.yandex.com/d/Io0siOESo2RAaA) -o data.zip

# или
wget -O data.zip $(yadisk-direct https://disk.yandex.com/d/Io0siOESo2RAaA)

#  и разархивировать
unzip -p data.zip train_ver2.csv > data.csv

# вернуться в каталог проекта
cd ..

# проверка FastAPI, endpoint "/predict", sample.json - файл с данными запроса
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d @models/sample.json

# проверка FastAPI, endpoint "/predict_proba"
curl -X POST "http://127.0.0.1:8000/predict_proba" \
     -H "Content-Type: application/json" \
     -d @models/sample.json

# сборка образов и старт контейнеров (~ 5 минут)
docker compose up --build
```
Интерфейсы сервисов  
FastAPI: http://127.0.0.1:8000   
MLFlow: http://127.0.0.1:5000   
Graphit: http://127.0.0.1:80   
Grafana: http://127.0.0.1:3000   
Airflow: http://127.0.0.1:8080   

В Grafana по умолчанию USER=admin, PASSWORD=grafana  
В UI выбрать Menu - Dashboards - bank-rs  

В Airflow по умолчанию USER=admin, PASSWORD=admin  
В UI выбрать DAGs -> sprint_6_airflow  

```
# в другом терминале:
# проверка запуска контейнеров:
docker ps

# диагностика запуска контейнеров (<service> - название сервиса ):
docker logs <service>

# проверка доступности портов
sudo lsof -i :8000

# запуск имитиации нагрузки на сервис (10000 запросов)
python run_mimic_load.py

# завершение работы сервисов по Ctrl-C в терминале, где был запущен контейнер
```

Опционально:
```
# Запуск отдельно FastAPI
uvicorn app:app --host 0.0.0.0 --port 8000

# Запуск отдельно FastAPI в Docker
docker build -t fastapi-app .
docker run -p 8000:8000 fastapi-app

### Запуск отдельно сервера MLFlow по скрипту ###
sh run_server.sh

### Запуск отдельно сервера Airflow в контейнере ###
# переход в рабочий каталог
cd airflow

# сборка образа в контексте родительского каталога
docker build -t airflow-standalone -f Dockerfile ../

# загрузка переменных окружения
source ../.env

# запуск сервиса
docker run -it --env-file ../.env -p ${AIRFLOW_PORT}:${AIRFLOW_PORT} airflow-standalone

### Очистка Docker ###
# удаление неиспользуемых образов
docker image prune -a

# удаление остановленных контейнеров
docker container prune

# удаление неиспользуемых томов
docker volume prune

# удаление ненужных кешей
docker builder prune

# полная очистка всех неиспользуемых ресурсов
docker system prune -a --volumes
``` 