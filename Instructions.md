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

# фиксация конкретных версий пакетов
pip install -r requirements.txt    

# в текущей папке должен быть файл .env со следующими переменными
DB_DESTINATION_*=
S3_*=
AWS_*=
MLFLOW_S3_ENDPOINT_URL=
TRACKING_SERVER_CONN=
DATA_DIR=data/
MODEL_DIR=models/
DATA_PARQUET=data.parquet
DATA_CSV=data.csv
TRAIN_PARQUET=df_train.parquet
TEST_PARQUET=df_test.parquet
MODEL_PKL=model.pkl
FITTED_MODEL=fitted_model.pkl
MODEL_PARAMS=model_params.json

# данные есть по прямой по ссылке
# если нет, установить unzip
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

# запуска сервера MLFlow по скрипту
sh run_server.sh
   
# проверка FastAPI, endpoint "/predict", sample.json - файл с данными запроса
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d @sample.json

# проверка FastAPI, endpoint "/predict_proba"
curl -X POST "http://127.0.0.1:8000/predict_proba" \
     -H "Content-Type: application/json" \
     -d @sample.json

# сборка образов и старт контейнеров
docker compose up --build

# интерфейсы сервисов
# FastAPI: http://127.0.0.1:8000 
# MLFlow: http://127.0.0.1:5000 
# Graphit: http://127.0.0.1:80 
# Grafana: http://127.0.0.1:3000 

# В Grafana по умолчанию USER=admin, PASSWORD=grafana
# В Grafana нужно выбрать Menu - Dashboards - bank-rs

# в другом терминале:
# проверка запуска контейнеров:
docker ps

# диагностика запуска контейнеров (<service> - название сервиса ):
docker logs <service>

# проверка доступности портов
sudo lsof -i :8000

# запуск имитиации нагрузки на сервис
python run_mimic_load.py


# завершение работы сервисов по Ctrl-C в терминале, где был запущен контейнер

# при необходимости: 
# удаление неиспользуемых образов
docker image prune -a

# удаление остановленных контейнеров
docker container prune

# удаление неиспользуемых томов
docker volume prune

# удаление ненужных кешей
docker builder prune

# полная очистка всех неипользуемых ресурсов
docker system prune -a --volumes







#pip install apache-airflow-providers-amazon

cd airflow
#docker build -t airflow-standalone .
docker build -t airflow-standalone -f Dockerfile ../

docker run -it --env-file ../.env -p ${AIRFLOW_PORT}:${AIRFLOW_PORT} airflow-standalone


``` 