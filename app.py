# Standard library imports
import os  # For environment variables
import json  # For parsing JSON files
import time  # For time tracking and delays
import numpy as np  # For array operations

# Third-party library imports
import uvicorn  # ASGI server for running FastAPI
import joblib  # For loading machine learning models
import psutil  # For accessing system utilization metrics
import pandas as pd  # For data manipulation and analysis
from fastapi import FastAPI, Body  # FastAPI components for building API
from dotenv import load_dotenv  # For loading environment variables from .env
from typing import Dict  # Type hinting for dictionaries
from statsd import StatsClient  # For sending metrics to StatsD

# Local project imports
from data_utils import (get_mem_usage, new_columns, attrs, DataFrameProcessor, 
                        frequency_encoding, get_X_y, gen_random_data)  # Custom utilities for data processing


# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
DATA_DIR = os.getenv('DATA_DIR')
MODEL_DIR = os.getenv('MODEL_DIR')
FITTED_MODEL = os.getenv('FITTED_MODEL')
MODEL_PARAMS = os.getenv('MODEL_PARAMS')


def load_row_from_json(json_data: Dict) -> pd.DataFrame:
    """
    Convert JSON data to a Pandas Series, handle date parsing, and return as a DataFrame.
    
    Args:
        json_data (Dict): JSON data received from the request.
    
    Returns:
        pd.DataFrame: DataFrame with a single row of parsed data.
    """
    row_loaded = pd.Series(json_data)

    # Parse date columns, ensuring consistent formatting and handling invalid dates
    row_loaded['fetch_date'] = pd.to_datetime(row_loaded['fetch_date'], format='%d-%m-%Y', errors='coerce')
    row_loaded['registration_date'] = pd.to_datetime(row_loaded['registration_date'], format='%d-%m-%Y', errors='coerce')
    row_loaded['last_date_as_primary'] = pd.to_datetime(row_loaded['last_date_as_primary'], format='%d-%m-%Y', errors='coerce')

    # Handle missing income values by replacing them with the mean income
    if pd.isna(row_loaded['income']):
        row_loaded['income'] = income_mean

    # Convert Series to DataFrame for model prediction compatibility
    return pd.DataFrame([row_loaded])


def interpret_predictions(predictions: np.ndarray, lang: str = 'rus', is_integer: bool = True) -> Dict[str, float]:
    """
    Map numeric predictions to target names in the specified language.
    
    Args:
        predictions (np.ndarray): Array of prediction values.
        lang (str): Language for target names, either 'rus' or 'eng'.
        is_integer (bool): Flag indicating whether to convert predictions to integers.
    
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


def refresh_metrics(st: StatsClient) -> None:
    """
    Update and send system metrics (CPU, memory usage, uptime) and request counts to StatsD.
    
    Args:
        st (StatsClient): StatsD client instance for sending metrics.
    """
    global last_time, service_start

    # Increment the request count metric
    st.incr('bank-rs.requests')
    
    # If the time delta exceeds the threshold, update system performance metrics
    if time.time() - last_time > time_delta:
        cpu_load = psutil.cpu_percent(interval=1)  # CPU load over 1 second interval
        st.gauge('bank-rs.system.cpu_load', cpu_load)
        memory_info = psutil.virtual_memory().available / 1024 / 1024  # Available memory in MB
        st.gauge('bank-rs.system.memory_free_mb', memory_info)
        last_time = time.time()

    # Send uptime metric (time since service started)
    st.gauge('bank-rs.system.up_time', time.time() - service_start)
    st.incr("response_code.200")  # Increment response code 200 metric


# Initialize global variables
service_start = time.time()  # Service start time (for uptime tracking)
last_time = time.time()  # Last time metrics were updated
time_delta = 10  # Time interval (in seconds) for metric refresh

# Load the pre-trained model from file
model = joblib.load(MODEL_DIR + FITTED_MODEL)

# Load model parameters (target names, income mean, etc.) from JSON file
with open(MODEL_DIR + MODEL_PARAMS, 'r') as f:
    json_data = json.load(f)
    target_names = json_data['target_names']  # Russian target names
    target_names_eng = json_data['target_names_eng']  # English target names
    income_mean = json_data['income_mean']  # Mean income used for missing values

# Initialize FastAPI app
app = FastAPI(title="Bank RS")

# Initialize StatsD client for sending metrics
stats_client = StatsClient(host="graphite", port=8125, prefix="bank-rs")


@app.get("/")
async def read_root() -> dict:
    """
    Root endpoint that returns the service status and refreshes system metrics.
    
    Returns:
        dict: Status message.
    """
    refresh_metrics(stats_client)
    return {"status": "Alive"}


@app.get("/random")
async def get_random() -> dict:
    """
    Generate random data, make predictions, and send metrics to StatsD.
    
    Returns:
        dict: Predicted values based on random data.
    """
    start_time = time.time()  # Track request start time

    # Generate random data and predict using the model
    row = load_row_from_json(gen_random_data())
    predictions = model.predict(row)[0]

    # Record response time and send to StatsD
    response_time = time.time() - start_time
    stats_client.timing("response_time", response_time)

    # Interpret predictions and increment StatsD counters for predicted targets
    predictions_dict = interpret_predictions(predictions, lang='eng')
    for target, prediction in predictions_dict.items():
        if prediction == 1:
            stats_client.incr("target." + target)

    # Refresh system metrics
    refresh_metrics(stats_client)

    # Return predictions as a dictionary
    return interpret_predictions(predictions)


@app.post("/predict")
async def predict(data: Dict = Body(...)) -> dict:
    """
    Predict based on provided input data and send metrics to StatsD.
    
    Args:
        data (Dict): Input data as JSON.
    
    Returns:
        dict: Predicted values.
    """
    start_time = time.time()  # Track request start time

    # Convert JSON data to DataFrame and predict using the model
    row = load_row_from_json(data)
    predictions = model.predict(row)[0]
    response_time = time.time() - start_time

    # Record response time and send to StatsD
    stats_client.timing("response_time", response_time)

    # Interpret predictions and increment StatsD counters for predicted targets
    predictions_dict = interpret_predictions(predictions, lang='eng')
    for target, prediction in predictions_dict.items():
        if prediction == 1:
            stats_client.incr("target." + target)

    # Refresh system metrics
    refresh_metrics(stats_client)

    # Return predictions as a dictionary
    return interpret_predictions(predictions)


@app.post("/predict_proba")
async def predict_proba(data: Dict = Body(...)) -> dict:
    """
    Makes recommendations based on probabilities, does not generate metrics.
    
    Args:
        data (Dict): Input data as JSON.
    
    Returns:
        dict: Sorted predictions with probabilities above the threshold.
    """
    row = load_row_from_json(data)
    predictions = model.predict_proba(row)

    threshold = 0.01
    interpreted_predictions = interpret_predictions([pos_class[1] for prediction in predictions for pos_class in prediction], is_integer=False)
    filtered_dict = {k: round(v, 4) for k, v in interpreted_predictions.items() if v >= threshold}
    sorted_dict = dict(sorted(filtered_dict.items(), key=lambda item: -item[1]))

    return sorted_dict
