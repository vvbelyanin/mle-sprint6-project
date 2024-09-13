README.md

**Инструкция по поднятию MLflow-сервисов и регистрации модели в MLflow Tracking Server:**    

В командной строке:    
```

# обновление установленных пакетов
sudo apt-get update    

# установка пакета виртуального окружения Python
sudo apt-get install python3.10-venv    

# создание виртуальной среды
python3.10 -m venv .venv_sprint_6  

# активирование окружения 
source .venv_sprint_6/bin/activate    

# фиксация конкретных версий пакетов
pip install -r requirements.txt    

# в текущей папке должен быть файл .env со следующими кредами
DB_DESTINATION_*, S3_*, AWS_*,
MLFLOW_S3_ENDPOINT_URL, TRACKING_SERVER_CONN

DATA_PARQUET=data/data.parquet
DATA_CSV=data/data.csv
TRAIN_PARQUET=data/df_train.parquet
TEST_PARQUET=data/df_test.parquet
MODEL_PKL=models/model.pkl
FITTED_MODEL=models/fitted_model.pkl
MODEL_PARAMS=models/model_params.json


# запуск сервера MLFlow
sh run_server.sh

# запуск кода загрузки данных и логирования в MLFlow
``` 
    
Параметры:    
    
S3 Bucket name: `s3-student-mle-20240325-4062b25c06`    
VS Code: 1.93.0


#!/bin/bash

# chmod +x run.sh
# ./run.sh

python3 -m pip install --upgrade pip
pip install wldhx.yadisk-direct
sudo apt install unzip

mkdir data && cd data

curl -L $(yadisk-direct https://disk.yandex.com/d/Io0siOESo2RAaA) -o data.zip
or
wget -O data.zip $(yadisk-direct https://disk.yandex.com/d/Io0siOESo2RAaA)

unzip -p data.zip train_ver2.csv > data.csv

cd ..

curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d @sample.json


docker pull grafana/grafana
docker pull graphiteapp/graphite-statsd

docker compose up --build


# check:
docker ps
docker logs fastapi_app


sudo lsof -i :8000

python run_mimic_load.py



Remove unused images: docker image prune -a
Remove stopped containers: docker container prune
Remove unused volumes: docker volume prune
Remove build cache: docker builder prune
Full cleanup: docker system prune -a --volumes