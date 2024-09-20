# Python file: app.py
#
# Content:
# Constants:
#    - REFRESH_PERIOD: Interval for refreshing system metrics.
#    - PROBA_THRESHOLD: Probability threshold for filtering predictions.
#
# Functions:
#    - refresh_metrics(st: StatsClient) -> None: Sends system metrics (CPU, memory) and request metrics to StatsD.
#
# FastAPI Endpoints:
#    - read_root() -> dict: Root endpoint that returns the status of the service.
#    - get_random() -> dict: Generates random data, makes predictions, and sends metrics to StatsD.
#    - predict(data: Dict = Body(...)) -> dict: Accepts user input data for predictions and sends metrics.
#    - predict_proba(data: Dict = Body(...)) -> dict: Returns sorted predictions with probabilities above a threshold.
#
# This file defines a FastAPI application with endpoints for generating predictions using a pre-trained model.
# It also includes system metric tracking via StatsD, and sends metrics related to CPU usage, memory, and API response times.

# Standard library imports
import os
import json
import time
import numpy as np

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
from utils.helpers import gen_random_data, load_row_from_json, interpret_predictions
from utils.config import (
    MODEL_DIR, FITTED_MODEL, STATSD_UDP_PORT, 
    REFRESH_PERIOD, PROBA_THRESHOLD, path
)


# Initialize global variables
service_start = time.time()  # Service start time (for uptime tracking)
last_time = time.time()  # Last time metrics were updated

# Load the pre-trained model from file
model = joblib.load(path(MODEL_DIR, FITTED_MODEL))

# Initialize FastAPI app
app = FastAPI(title="Bank RS")

# Initialize StatsD client for sending metrics
stats_client = StatsClient(host="graphite", port=STATSD_UDP_PORT, prefix="bank-rs")


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
    if time.time() - last_time > REFRESH_PERIOD:
        cpu_load = psutil.cpu_percent(interval=1) / 100  # CPU load over 1 second interval
        st.gauge('bank-rs.system.cpu_load', cpu_load)
        memory_info = psutil.virtual_memory().available / 1024 / 1024  # Available memory in MB
        st.gauge('bank-rs.system.memory_free_mb', memory_info)
        last_time = time.time()

    # Send uptime metric (time since service started)
    st.gauge('bank-rs.system.up_time', time.time() - service_start)
    st.incr("response_code.200")  # Increment response code 200 metric


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

    interpreted_predictions = interpret_predictions([pos_class[1] for prediction in predictions for pos_class in prediction], is_integer=False)
    filtered_dict = {k: round(v, 4) for k, v in interpreted_predictions.items() if v >= PROBA_THRESHOLD}
    sorted_dict = dict(sorted(filtered_dict.items(), key=lambda item: -item[1]))

    return sorted_dict
