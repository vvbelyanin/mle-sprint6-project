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


curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
    "fetch_date":"28-04-2016",
    "id":1251582,
    "ind_employee":"N",
    "country_of_residence":"ES",
    "gender":"H",
    "age":" 30",
    "registration_date":"13-04-2016",
    "ind_new_client":0.0,
    "tenure_months":"     24",
    "client_relationship_status":1.0,
    "last_date_as_primary":null,
    "client_type_1m":"2",
    "client_activity_1m":"A",
    "ind_resident":"S",
    "ind_foreigner":"N",
    "ind_spouse_employee":null,
    "entry_channel":"KFC",
    "ind_deceased":"N",
    "address_type":1.0,
    "province_code":28.0,
    "province_name":"MADRID",
    "ind_client_activity":0.0,
    "income":null,
    "client_segment":null,
    "ind_1m_savings_acc":0,
    "ind_1m_guarantee":0,
    "ind_1m_checking_acc":0,
    "ind_1m_derivatives":0,
    "ind_1m_payroll_acc":0,
    "ind_1m_junior_acc":0,
    "ind_1m_mature_acc_3":0,
    "ind_1m_operations_acc":0,
    "ind_1m_pension_acc_2":0,
    "ind_1m_short_term_deposit":0,
    "ind_1m_medium_term_deposit":0,
    "ind_1m_long_term_deposit":0,
    "ind_1m_digital_account":0,
    "ind_1m_cash_funds":0,
    "ind_1m_mortgage":0,
    "ind_1m_pension_plan":0,
    "ind_1m_loans":0,
    "ind_1m_tax_account":0,
    "ind_1m_credit_card":0,
    "ind_1m_securities":0,
    "ind_1m_home_acc":0,
    "ind_1m_salary_acc":0.0,
    "ind_1m_pension_obligation_account":0.0,
    "ind_1m_debit_account":0
}'
