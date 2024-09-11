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
DB_DESTINATION_HOST=<...>
DB_DESTINATION_PORT=<...>
DB_DESTINATION_NAME=<...>
DB_DESTINATION_USER=<...>
DB_DESTINATION_PASSWORD=<...>
S3_BUCKET_NAME=<...>
AWS_ACCESS_KEY_ID=<...>
AWS_SECRET_ACCESS_KEY=<...>
MLFLOW_S3_ENDPOINT_URL=<...>
TRACKING_SERVER_CONN=http://127.0.0.1:5000

# загрузка переменных окружения из файла .env
export $(cat .env | xargs)    

# запуск сервера MLFlow
sh run_server.sh

# запуск кода загрузки данных и логирования в MLFlow
cd mlflow_server
python3 run_experiment.py
``` 
    
Параметры:    
    
S3 Bucket name: `s3-student-mle-20240325-4062b25c06`    
MLFlow experiment name:  `Спринт 3/9: 2 спринт → Тема 5/5: Проект`    
MLFlow run name: `ETL`    
TRACKING_SERVER_CONN=http://127.0.0.1:5000