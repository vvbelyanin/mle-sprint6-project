# Python file: utils/config.py
#
# Content:
#
# Constants:
#    - S3_BUCKET_NAME: S3 bucket name for storage.
#    - AWS_ACCESS_KEY_ID: AWS access key for S3 operations.
#    - AWS_SECRET_ACCESS_KEY: AWS secret access key for S3.
#    - S3_ENDPOINT_URL: Custom endpoint URL for S3.
#    - STATSD_UDP_PORT: Port for StatsD monitoring.
#    - AWS_CONN_ID: AWS connection ID for specific integrations.
#
# Date Constants:
#    - TRAIN_TEST_SPLIT_DATE: Specifies the date to split training and testing data.
#    - START_TRAIN_DATE: Specifies the start date for the training dataset.
#
# Directory Paths:
#    - DATA_DIR: Path to the directory where raw data is stored.
#    - MODEL_DIR: Path to the directory where models are saved.
#    - TMP_DIR: Path to the directory for temporary files.
#    - S3_DIR: Path to the directory in S3 for data storage.
#
# File Names:
#    - DATA_ZIP: Name of the ZIP file containing the raw data.
#    - DATA_CSV: Name of the CSV file containing the raw data.
#    - DATA_PARQUET: Name of the Parquet file for data processing.
#    - TRAIN_PARQUET: Name of the Parquet file for the training dataset.
#    - TEST_PARQUET: Name of the Parquet file for the test dataset.
#    - MODEL_PKL: Name of the file where the model is saved in pickle format.
#    - FITTED_MODEL: Name of the file for storing the fitted model.
#    - MODEL_PARAMS: Name of the JSON file where model parameters are stored.
#    - SAMPLE_JSON: Name of the sample JSON file.
#
# Functions:
#    - path(base: str, *parts: str) -> str: Joins a base path with additional parts to create a full file path.
#
# Model Parameters:
#    - target_names: List of target variable names in Russian.
#    - target_names_eng: List of target variable names in English.
#    - income_mean: Mean income value, used for handling missing income data.
#
# Data Type Specifications:
#    - dtype_spec: Specifies the expected data types for certain columns for parsing csv.
#
# Column Definitions:
#    - date_columns: List of columns that represent date values.
#    - numerical_columns: List of columns containing numerical values.
#    - freq_encode_columns: List of columns to be frequency-encoded.
#    - one_hot_columns: List of columns for one-hot encoding.
#
# Column Mappings:
#    - new_columns: Mapping of old column names to new, standardized names for consistency in the dataset.
#
# Column Attributes:
#    - attrs: Describes attributes and the meaning of each column, including user demographic data, product features, and income data.


from dotenv import load_dotenv
import os
import json
from typing import Dict, Tuple, Union

# Load environment variables from a .env file
load_dotenv()

# Load AWS-related environment variables
S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME")
AWS_ACCESS_KEY_ID: Union[str, None] = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY: Union[str, None] = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_ENDPOINT_URL: Union[str, None] = os.getenv('S3_ENDPOINT_URL')

# Load connection information
STATSD_UDP_PORT: Union[str, None] = os.getenv('STATSD_UDP_PORT')
AWS_CONN_ID: Union[str, None] = os.getenv('AWS_CONN_ID')
TRACKING_SERVER_CONN = os.getenv('TRACKING_SERVER_CONN')

# Define date constants
TRAIN_TEST_SPLIT_DATE: str = '2016-02-28'
START_TRAIN_DATE: str = '2015-07-28'

# Define directory paths
DATA_DIR: str = 'data/'
MODEL_DIR: str = 'models/'
TMP_DIR: str = 'tmp/'
S3_DIR: str = 'recsys/sprint_6_data/'

# Define file names
DATA_ZIP: str = 'data.zip'
DATA_CSV: str = 'data.csv'
DATA_PARQUET: str = 'data.parquet'
TRAIN_PARQUET: str = 'df_train.parquet'
TEST_PARQUET: str = 'df_test.parquet'

MODEL_PKL: str = 'model.pkl'
FITTED_MODEL: str = 'fitted_model.pkl'
MODEL_PARAMS: str = 'model_params.json'
SAMPLE_JSON: str = 'sample.json'

LUCKY_USER = 376088 # Sample user for predictions testing

RANDOM_STATE = 42  # Random state for reproducibility
NAN_THRESHOLD = 10  # Threshold for NaN values to drop rows

CBC_ITERATIONS = 100  # Number of iterations for CatBoostClassifier
CBC_VERBOSE = 100  # Verbosity level for CatBoost
CBC_CLASS_WEIGHTS = [1, 4]  # Class weights for imbalanced data

# Constants
REFRESH_PERIOD = 10  # Time interval (in seconds) for metric refresh
PROBA_THRESHOLD = 0.01  # Probability threshold for filtering predictions


def path(base: str, *parts: str) -> str:
    """
    Joins base path with additional parts.

    Args:
        base (str): The base directory.
        *parts (str): Additional directory or file parts.

    Returns:
        str: The full path formed by joining the base with the parts.
    """
    return os.path.join(base, *parts)


# Load model parameters from a JSON file
model_params_path = path(MODEL_DIR, MODEL_PARAMS)

if os.path.exists(model_params_path):
    with open(model_params_path, 'r') as f:
        json_data: dict = json.load(f)
        target_names: list = json_data['target_names']
        target_names_eng: list = json_data['target_names_eng']
        income_mean: float = json_data['income_mean']
else:
    target_names = []
    target_names_eng = []
    income_mean = 0.0

# Specify the data types for certain columns for parsing csv
dtype_spec: Dict[str, str] = {
    'age': 'str',
    'tenure_months': 'str',
    'client_type_1m': 'str',
    'ind_spouse_employee': 'str',
}

# Define various columns by type
date_columns: list = ['fetch_date', 'registration_date', 'last_date_as_primary']
numerical_columns: list = ['age', 'tenure_months', 'income', 'fetch_year', 'fetch_month', 'number_of_products']
freq_encode_columns: list = ['entry_channel', 'province_code']
one_hot_columns: list = ['gender', 'client_type_1m', 'client_activity_1m', 'ind_foreigner', 'client_segment']

# Mapping of old column names to new column names
new_columns: Dict[str, str] = {
    'fecha_dato': 'fetch_date',
    'ncodpers': 'id',
    'ind_empleado': 'ind_employee',
    'pais_residencia': 'country_of_residence',
    'sexo': 'gender',
    'age': 'age',
    'fecha_alta': 'registration_date',
    'ind_nuevo': 'ind_new_client',
    'antiguedad': 'tenure_months',
    'indrel': 'client_relationship_status',
    'ult_fec_cli_1t': 'last_date_as_primary',
    'indrel_1mes': 'client_type_1m',
    'tiprel_1mes': 'client_activity_1m',
    'indresi': 'ind_resident',
    'indext': 'ind_foreigner',
    'conyuemp': 'ind_spouse_employee',
    'canal_entrada': 'entry_channel',
    'indfall': 'ind_deceased',
    'tipodom': 'address_type',
    'cod_prov': 'province_code',
    'nomprov': 'province_name',
    'ind_actividad_cliente': 'ind_client_activity',
    'renta': 'income',
    'segmento': 'client_segment',
    'ind_ahor_fin_ult1': 'ind_1m_savings_acc',
    'ind_aval_fin_ult1': 'ind_1m_guarantee',
    'ind_cco_fin_ult1': 'ind_1m_checking_acc',
    'ind_cder_fin_ult1': 'ind_1m_derivatives',
    'ind_cno_fin_ult1': 'ind_1m_payroll_acc',
    'ind_ctju_fin_ult1': 'ind_1m_junior_acc',
    'ind_ctma_fin_ult1': 'ind_1m_mature_acc_3',
    'ind_ctop_fin_ult1': 'ind_1m_operations_acc',
    'ind_ctpp_fin_ult1': 'ind_1m_pension_acc_2',
    'ind_deco_fin_ult1': 'ind_1m_short_term_deposit',
    'ind_deme_fin_ult1': 'ind_1m_medium_term_deposit',
    'ind_dela_fin_ult1': 'ind_1m_long_term_deposit',
    'ind_ecue_fin_ult1': 'ind_1m_digital_account',
    'ind_fond_fin_ult1': 'ind_1m_cash_funds',
    'ind_hip_fin_ult1': 'ind_1m_mortgage',
    'ind_plan_fin_ult1': 'ind_1m_pension_plan',
    'ind_pres_fin_ult1': 'ind_1m_loans',
    'ind_reca_fin_ult1': 'ind_1m_tax_account',
    'ind_tjcr_fin_ult1': 'ind_1m_credit_card',
    'ind_valo_fin_ult1': 'ind_1m_securities',
    'ind_viv_fin_ult1': 'ind_1m_home_acc',
    'ind_nomina_ult1': 'ind_1m_salary_acc',
    'ind_nom_pens_ult1': 'ind_1m_pension_obligation_account',
    'ind_recibo_ult1': 'ind_1m_debit_account'
}

# Define the attributes and descriptions for each column
attrs: Dict[str, Tuple[str, Union[Dict[str, str], str]]] = {
    'fetch_date': ('Колонка для разделения таблицы', {}),
    'id': ('Идентификатор пользователя', {}),
    'ind_employee': ('Статус занятости', {
        'A': 'трудоустроен',
        'B': 'безработный, раньше работал', 
        'F': 'иждивенец',
        'N': 'безработный',
        'P': 'пассивный (статус не определён)'
    }),
    'country_of_residence': ('Страна резидентства', {}),
    'gender': ('Пол', {}), 
    'age': ('Возраст', {}),
    'registration_date': ('Дата, когда клиент впервые заключил договор в банке', {}),
    'ind_new_client': ('Признак нового клиента', {
        '1': 'если клиент зарегистрировался за последние 6 месяцев'
    }),
    'tenure_months': ('Стаж клиента (в месяцах)', {}), 
    'client_relationship_status': ('Признак первичного клиента', {
        '1': 'первичный клиент', 
        '99': 'первичный клиент в течении месяца, но не в конце'
    }),
    'last_date_as_primary': ('Последняя дата, когда клиент был премиальным', {}),
    'client_type_1m': ('Тип клиента в начале месяца', {
        '1': 'премиальный',
        '2': 'собственник', 
        'P': 'потенциальный', 
        '3': 'раньше был премиальным', 
        '4': 'раньше был собственником'
    }),
    'client_activity_1m': ('Тип клиента в начале месяца', {
        'A': 'активный', 
        'I': 'неактивный', 
        'P': 'бывший',
        'R': 'потенциальный'
    }), 
    'ind_resident': ('Если страна проживания совпадает со страной банка', {}),
    'ind_foreigner': ('Если страна рождения клиента отличается от страны банка', {}),
    'ind_spouse_employee': ('Признак супруга работника', {'1': 'если клиент супруг(а) работника'}),
    'entry_channel': ('Канал, по которому пришел пользователь', {}),
    'ind_deceased': ('Индекс актуальности счёта (англ. Deceased index, N/S)', {}),
    'address_type': ('Тип адреса', {'1': 'основной адрес'}),
    'province_code': ('Код провинции (адреса клиента)', {}),
    'province_name': ('Имя провинции', {}),
    'ind_client_activity': ('Активность пользователя', {
        '1': 'активный',
        '0': 'неактивный'
    }),
    'income': ('Доход домохозяйства', {}),
    'client_segment': ('Сегментация', {
        '1': 'VIP',
        '2': 'Обыкновенные',
        '3': 'Выпускники колледжей'
    }),
    'ind_1m_savings_acc': ('Сберегательный счёт', 'Savings Account'),
    'ind_1m_guarantee': ('Банковская гарантия', 'Bank Guarantee'),
    'ind_1m_checking_acc': ('Текущие счета', 'Checking Account'),
    'ind_1m_derivatives': ('Деривативный счёт', 'Derivatives Account'),
    'ind_1m_payroll_acc': ('Зарплатный проект', 'Payroll Account'),
    'ind_1m_junior_acc': ('Детский счёт', 'Junior Account'),
    'ind_1m_mature_acc_3': ('Особый счёт 3', 'Special Account 3'),
    'ind_1m_operations_acc': ('Особый счёт', 'Operations Account'),
    'ind_1m_pension_acc_2': ('Особый счёт 2', 'Special Account 2'),
    'ind_1m_short_term_deposit': ('Краткосрочный депозит', 'Short-Term Deposit'),
    'ind_1m_medium_term_deposit': ('Среднесрочный депозит', 'Medium-Term Deposit'),
    'ind_1m_long_term_deposit': ('Долгосрочный депозит', 'Long-Term Deposit'),
    'ind_1m_digital_account': ('Цифровой счёт', 'Digital Account'),
    'ind_1m_cash_funds': ('Денежный средства', 'Cash Funds'),
    'ind_1m_mortgage': ('Ипотека', 'Mortgage'),
    'ind_1m_pension_plan': ('Пенсионный план', 'Pension Plan'),
    'ind_1m_loans': ('Кредит', 'Loans'),
    'ind_1m_tax_account': ('Налоговый счёт', 'Tax Account'),
    'ind_1m_credit_card': ('Кредитная карта', 'Credit Card'),
    'ind_1m_securities': ('Ценные бумаги', 'Securities'),
    'ind_1m_home_acc': ('Домашний счёт', 'Home Account'),
    'ind_1m_salary_acc': ('Аккаунт для выплаты зарплаты', 'Salary Account'),
    'ind_1m_pension_obligation_account': (
        'Аккаунт для пенсионных обязательств', 'Pension Obligation Account'
    ),
    'ind_1m_debit_account': ('Дебетовый аккаунт', 'Debit Account')
}
