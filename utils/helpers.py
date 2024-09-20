# Python file: data_utility_functions.py
#
# Content:
# 1. Imports:
#    - Standard libraries for system interaction, random data generation, date handling, and type hinting.
#      * sys: For system-specific parameters and functions.
#      * random: For random number generation.
#      * datetime: For date and time manipulation.
#      * typing: For type hints (Optional, Dict, Any).
#    - Third-party libraries for data manipulation, system monitoring, and file handling.
#      * pandas (pd): Data manipulation and analysis.
#      * psutil: For system memory and CPU statistics.
#      * humanize: For human-readable memory size conversion.
#      * json: For reading and writing JSON data.
#      * os: For interacting with the operating system.
#      * numpy (np): For numerical operations.
#    - Configuration imports for column handling and target names.
#      * utils.config: For loading date columns and target names (in Russian and English) and mean income handling.
#
# 2. Functions:
#    - interpret_predictions(predictions: np.ndarray, lang: str, is_integer: bool) -> Dict[str, float]: 
#      Maps numeric predictions to target names in the specified language ('rus' or 'eng').
#    - load_row_from_json(json_data: Dict[str, Any]) -> pd.DataFrame:
#      Converts JSON data into a Pandas DataFrame, handling date columns and missing income.
#    - load_json(filename: str) -> Any:
#      Loads a JSON object from a file.
#    - save_json(obj: Any, filename: str) -> None:
#      Saves a Python object as a JSON file.
#    - add_fields_overrides_to_grafana(filename: str, attrs: Dict[str, tuple]) -> None:
#      Adds field overrides to a Grafana JSON configuration file based on provided attributes.
#    - gen_random_data(output: str = 'json') -> Dict[str, Any]:
#      Generates random user data in either JSON or DataFrame format for testing.
#    - get_mem_usage(top_k: Optional[int] = None) -> None:
#      Displays memory usage statistics, optionally showing the top-k largest variables in the environment.
#
# This file contains utility functions for data manipulation, prediction interpretation, system monitoring, 
# random data generation, and interaction with JSON and Grafana configuration files.


# Standard library imports
import sys
import random
from datetime import datetime
from typing import Optional, Dict, Any

# Third-party library imports
import pandas as pd
import psutil
from humanize import naturalsize
import json
import os
import numpy as np

# Import the list of date-related columns that require special handling (e.g., parsing, formatting)
from utils.config import date_columns

# Import target names in Russian and English for prediction interpretation, and the mean income for handling missing values
from utils.config import target_names, target_names_eng, income_mean


def interpret_predictions(predictions: np.ndarray, lang: str = 'rus', is_integer: bool = True) -> Dict[str, float]:
    """
    Map numeric predictions to target names in the specified language.

    Args:
        predictions (np.ndarray): Array of prediction values.
        lang (str): Language for target names, either 'rus' or 'eng'. Defaults to 'rus'.
        is_integer (bool): Flag indicating whether to convert predictions to integers. Defaults to True.

    Returns:
        Dict[str, float]: Dictionary mapping target names to their predicted values.
    """
    if lang == 'rus':
        targets = target_names
    else:
        targets = target_names_eng

    # Construct a dictionary mapping target names to prediction values
    res = {}
    for col, name in zip(predictions, targets):
        if is_integer:
            res[name] = int(col)  # Convert to int if needed
        else:
            res[name] = float(col)  # Convert to float if needed

    return res


def load_row_from_json(json_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert JSON data to a Pandas Series, handle date parsing, and return as a DataFrame.

    Args:
        json_data (Dict[str, Any]): JSON data received from the request.

    Returns:
        pd.DataFrame: DataFrame with a single row of parsed data.
    """
    row_loaded = pd.Series(json_data)

    # Parse date columns, ensuring consistent formatting and handling invalid dates
    row_loaded[date_columns] = row_loaded[date_columns].apply(lambda col: pd.to_datetime(col, format='%d-%m-%Y', errors='coerce'))

    # Handle missing income values by replacing them with the mean income
    if pd.isna(row_loaded['income']):
        row_loaded['income'] = income_mean

    # Convert Series to DataFrame for model prediction compatibility
    return pd.DataFrame([row_loaded])


def load_json(filename: str) -> Any:
    """
    Load a JSON object from a file.

    Args:
        filename (str): Path to the JSON file.

    Returns:
        Any: The loaded JSON object.
    """
    with open(filename, 'r') as f:
        return json.load(f)


def save_json(obj: Any, filename: str) -> None:
    """
    Save a JSON object to a file.

    Args:
        obj (Any): The JSON-serializable object to save.
        filename (str): Path to the file to save the JSON object.
    """
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=4)


def add_fields_overrides_to_grafana(filename: str, attrs: Dict[str, tuple]) -> None:
    """
    Add field overrides to a Grafana JSON configuration file.

    Args:
        filename (str): Path to the Grafana configuration file.
        attrs (Dict[str, tuple]): Attributes to be overridden in the Grafana dashboard.
    """
    gr_dict = load_json(filename)

    for k, v in attrs.items():
        if k.startswith('ind_1m_'):
            override_dict = {
                'matcher': {'id': 'byRegexp', 'options': '.*' + v[1].replace(' ', '_') + '.*'},
                'properties': [{'id': 'displayName', 'value': v[0]}]
            }
            gr_dict['panels'][0]['fieldConfig']['overrides'].append(override_dict)

    save_json(gr_dict, filename)


def gen_random_data(output: str = 'json') -> Dict[str, Any]:
    """
    Generates a dictionary of random data with specific fields.

    Args:
        output (str): The format of the output ('json' for JSON, default is 'json').

    Returns:
        Dict[str, Any]: A dictionary with randomly generated values for specified fields.
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
        # Use .apply for element-wise operations across columns
        data_df[date_columns] = data_df[date_columns].apply(lambda col: col.map(lambda x: x.strftime('%d-%m-%Y')))
        return data_df.to_dict(orient='index')[0]  # Return first (and only) row as dict

    return data_df


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
