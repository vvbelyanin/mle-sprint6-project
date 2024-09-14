# Standard library imports
import os
import json
import time

# Third-party library imports
import uvicorn
import joblib
import psutil
import pandas as pd
from fastapi import FastAPI, Body
from dotenv import load_dotenv
from typing import Dict
from statsd import StatsClient

# Local project imports
from data_utils import (get_mem_usage, new_columns, attrs, DataFrameProcessor, 
                        frequency_encoding, get_X_y, gen_random_data)


# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
DATA_DIR = os.getenv('DATA_DIR')
MODEL_DIR = os.getenv('MODEL_DIR')
FITTED_MODEL = os.getenv('FITTED_MODEL')
MODEL_PARAMS = os.getenv('MODEL_PARAMS')

def load_row_from_json(json_data):
    """
    Convert JSON data to a Pandas Series, handle date parsing, and return as a DataFrame.
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

def interpret_predictions(predictions, lang='rus'):
    """
    Map numeric predictions to target names in the specified language.
    """
    if lang == 'rus':
        targets = target_names
    else:
        targets = target_names_eng
    
    # Construct a dictionary mapping target names to prediction values
    res = {}
    for col, name in zip(predictions, targets):
        res[name] = int(col)
    return res

def refresh_metrics(st):
    """
    Update and send system metrics (CPU, memory usage, uptime) and request counts to StatsD.
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
    """
    refresh_metrics(stats_client)
    return {"status": "Alive"}

@app.get("/random")
async def get_random() -> dict:
    """
    Generate random data, make predictions, and send metrics to StatsD.
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
async def predict(data: Dict = Body(...)):
    """
    Predict based on provided input data and send metrics to StatsD.
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
