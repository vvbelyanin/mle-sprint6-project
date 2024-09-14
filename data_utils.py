# Standard library imports
import sys
import os
import random
from datetime import datetime
from typing import Optional, Dict, Any

# Third-party library imports
import pandas as pd
import numpy as np
import psutil
from humanize import naturalsize
import boto3
from botocore.exceptions import ClientError, PartialCredentialsError
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin


def gen_random_data(output: str = 'json') -> Dict[str, Any]:
    """
    Generates a dictionary of random data with specific fields.

    Args:
        output: The format of the output ('json' for JSON, default is 'json').

    Returns:
        A dictionary with randomly generated values for specified fields.
    """
    data = {
        "fetch_date": random.choice(
            [datetime(random.choice([2015, 2016]), month, 28).strftime("%d-%m-%Y") for month in range(1, 13)]
        ),
        "id": random.randint(1, 1300000),
        "ind_employee": random.choice(['A', 'B', 'F', 'N', 'P']),
        "country_of_residence": "ES",
        "gender": random.choice(["H", "V"]),
        "age": str(random.randint(1, 100)),
        "registration_date": random.choice(
            [datetime(random.choice(range(2000, 2016)), month, 1).strftime("%d-%m-%Y") for month in range(1, 13)]
        ),
        "ind_new_client": random.choice([0, 1]),
        "tenure_months": str(random.randint(1, 264)),
        "client_relationship_status": random.choice([1.0, 99.0]),
        "last_date_as_primary": random.choice(
            [datetime(random.choice(range(2015, 2016)), month, 1).strftime("%d-%m-%Y") for month in range(1, 13)]
        ),
        "client_type_1m": random.choice(['1', '2', '3', '4', 'P']),
        "client_activity_1m": random.choice(['A', 'I', 'P', 'R']),
        "ind_resident": random.choice(['S', 'N']),
        "ind_foreigner": random.choice(['S', 'N']),
        "ind_spouse_employee": random.choice(['S', 'N']),
        "entry_channel": "KFC",
        "ind_deceased": random.choice(['S', 'N']),
        "address_type": 1.0,
        "province_code": 28.0,
        "province_name": "MADRID",
        "ind_client_activity": random.choice(['0', '1']),
        "income": random.randint(10000, 500000),
        "client_segment": random.choice(['02 - PARTICULARES', '03 - UNIVERSITARIO', '01 - TOP']),
        "ind_1m_savings_acc": random.choice([0, 1]),
        "ind_1m_guarantee": random.choice([0, 1]),
        "ind_1m_checking_acc": random.choice([0, 1]),
        "ind_1m_derivatives": random.choice([0, 1]),
        "ind_1m_payroll_acc": random.choice([0, 1]),
        "ind_1m_junior_acc": random.choice([0, 1]),
        "ind_1m_mature_acc_3": random.choice([0, 1]),
        "ind_1m_operations_acc": random.choice([0, 1]),
        "ind_1m_pension_acc_2": random.choice([0, 1]),
        "ind_1m_short_term_deposit": random.choice([0, 1]),
        "ind_1m_medium_term_deposit": random.choice([0, 1]),
        "ind_1m_long_term_deposit": random.choice([0, 1]),
        "ind_1m_digital_account": random.choice([0, 1]),
        "ind_1m_cash_funds": random.choice([0, 1]),
        "ind_1m_mortgage": random.choice([0, 1]),
        "ind_1m_pension_plan": random.choice([0, 1]),
        "ind_1m_loans": random.choice([0, 1]),
        "ind_1m_tax_account": random.choice([0, 1]),
        "ind_1m_credit_card": random.choice([0, 1]),
        "ind_1m_securities": random.choice([0, 1]),
        "ind_1m_home_acc": random.choice([0, 1]),
        "ind_1m_salary_acc": random.choice([0, 1]),
        "ind_1m_pension_obligation_account": random.choice([0, 1]),
        "ind_1m_debit_account": random.choice([0, 1])
    }

    data_df = pd.DataFrame([data])
    cols_to_date = ["fetch_date", "registration_date", "last_date_as_primary"]
    data_df[cols_to_date] = data_df[cols_to_date].astype('datetime64[ns]')

    if output == 'json':
        data_df['fetch_date'] = data_df['fetch_date'].dt.strftime('%d-%m-%Y')
        data_df['registration_date'] = data_df['registration_date'].dt.strftime('%d-%m-%Y')
        data_df['last_date_as_primary'] = data_df['last_date_as_primary'].dt.strftime('%d-%m-%Y')

        return data_df.to_dict(orient='index')[0]  # Return first (and only) row as dict

    return data_df


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DataFrame by adding calculated features and cleaning up data.

    Args:
        df: The DataFrame to process.

    Returns:
        Processed DataFrame with new columns and cleaned data.
    """
    df['number_of_products'] = df[[col for col in df.columns if col.startswith('ind_1m_')]].sum(axis=1)
    df['fetch_year'] = df['fetch_date'].dt.year
    df['fetch_month'] = df['fetch_date'].dt.month

    df['tenure_months'] = (
        (df['fetch_date'].dt.year - df['registration_date'].dt.year) * 12
        + (df['fetch_date'].dt.month - df['registration_date'].dt.month)
    )

    df['income'] = np.log(df['income'])
    df['client_type_1m'] = pd.to_numeric(df['client_type_1m'], errors='coerce').replace('P', 5)
    df['client_segment'] = df['client_segment'].map({
        None: 0,
        '02 - PARTICULARES': 2,
        '03 - UNIVERSITARIO': 3,
        '01 - TOP': 1
    })

    df.fillna({
        'province_code': 0,
        'gender': 'V',
        'client_activity_1m': 'N',
        'entry_channel': 'UNK',
        'income': df['income'].median(),
        'client_type_1m': 0
    }, inplace=True)

    columns_to_int = [
        'age', 'tenure_months', 'ind_new_client', 'client_relationship_status',
        'ind_client_activity', 'province_code', 'client_type_1m', 'number_of_products'
    ]
    df[columns_to_int] = df[columns_to_int].astype(int)

    df = df.drop(
        columns=[
            'fetch_date', 'id', 'ind_deceased', 'ind_spouse_employee', 'last_date_as_primary',
            'address_type', 'ind_employee', 'country_of_residence', 'ind_resident', 'province_name', 'registration_date'
        ]
    )

    return df.fillna(0)


def frequency_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features in a DataFrame with their frequency.

    Args:
        X: DataFrame to encode.

    Returns:
        A new DataFrame where each categorical feature is replaced by its frequency.
    """
    X_copy = X.copy()
    for col in X_copy.columns:
        freq_map = X_copy[col].value_counts(normalize=True).to_dict()
        X_copy[col] = X_copy[col].map(freq_map)
    return X_copy


class DataFrameProcessor(BaseEstimator, TransformerMixin):
    """
    A custom transformer that processes a DataFrame.
    """
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> 'DataFrameProcessor':
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        processed_X = process_df(X.copy())
        if y is not None:
            processed_y = y.loc[processed_X.index]
            return processed_X, processed_y
        return processed_X


def get_X_y(df: pd.DataFrame, targets: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into features (X) and targets (y).

    Args:
        df: DataFrame to process.
        targets: List of columns to be used as targets.

    Returns:
        Tuple of (X, y) where X is the DataFrame of features and y is the DataFrame of targets.
    """
    df = df[df.isna().sum(axis=1) < 10]
    df = df[df['ind_deceased'] == 'N']
    return df.drop(columns=targets), df[targets]


def get_mem_usage(top_k: Optional[int] = None) -> None:
    """
    Display memory usage statistics, including the top-k largest variables 
    in the current environment and system memory stats.

    Args:
        top_k (Optional[int]): The number of top variables to display, sorted by size.
                               If None, the system memory stats are displayed only.
    """
    if top_k:
        print(f"Топ-{top_k} объемных переменных:")
        # Get the size of each global variable and display the top_k largest
        for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()), key=lambda x: -x[1])[:top_k]:
            print(f"{name}: {naturalsize(size)}")
        print()

    memory = psutil.virtual_memory()
    print(f"Общая память: {naturalsize(memory.total)}")
    print(f"Доступная память: {naturalsize(memory.available)}")


def show_s3_folder(path: str) -> None:
    """
    List all files in a specified S3 folder path.

    Args:
        path (str): The S3 folder path to list the contents from.
    """
    s3_client = boto3.client(
        's3', 
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), 
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('S3_ENDPOINT_URL')
    )

    # Paginate through the files in the specified folder
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=os.getenv("S3_BUCKET_NAME"), Prefix=path)

    print(f"Files in {path} folder:")
    counter = 0
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if not key.endswith('/'):
                    relative_path = key[len(path):]
                    if '/' not in relative_path:
                        print(key)
                        counter += 1
    if counter == 0:
        print("No files found.")
    else:
        print(f"{counter} file(s) found.")


def download_from_s3(file_name: str, object_name: str = None) -> Any:
    """
    Download a file from an S3 bucket and return an appropriate object depending on the file type.

    Args:
        file_name (str): The name of the file to save locally.
        object_name (str, optional): The name of the object in the S3 bucket. Defaults to file_name if not provided.

    Returns:
        Any: Loaded Parquet file or CatBoost model based on the file type.
    """
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('S3_ENDPOINT_URL')
    )

    s3_client.download_file(os.getenv("S3_BUCKET_NAME"), object_name, file_name)

    if file_name.endswith('.parquet'):
        return pd.read_parquet(file_name)
    elif file_name.endswith('.cbm'):
        model = CatBoostClassifier()
        model.load_model(file_name)
        return model
    else:
        raise ValueError("Unsupported file type. Only .parquet and .cbm files are supported.")


def upload_to_s3(file_name: str, object_name: str = None) -> bool:
    """
    Upload a file to an S3 bucket.

    Args:
        file_name (str): The local file path to upload.
        object_name (str, optional): The object name in the S3 bucket. Defaults to file_name if not provided.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('S3_ENDPOINT_URL')
    )

    try:
        s3_client.upload_file(file_name, os.getenv("S3_BUCKET_NAME"), object_name)
        print(f"File {file_name} uploaded to {object_name}")
        return True
    except PartialCredentialsError:
        print("Incomplete credentials provided.")
    except ClientError as e:
        print(f"Upload failed:\nDetails: {str(e)}")
    except Exception as e:
        print(f"General Error: An unexpected error occurred: {str(e)}")
    return False


new_columns = {
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


attrs = {
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
        '1' : 'если клиент зарегистрировался за последние 6 месяцев'
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
    'ind_1m_pension_obligation_account': ('Аккаунт для пенсионных обязательств', 'Pension Obligation Account'),
    'ind_1m_debit_account': ('Дебетовый аккаунт', 'Debit Account')
}