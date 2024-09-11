"""
services/app/app-stage-4.py

This module adds enhanced monitoring and error simulation to the FastAPI application.

1. Imports the FastAPI application and instrumentation from previous stages.
2. Defines additional Prometheus metrics for CPU, disk, memory, and network usage.
3. Replaces the original "/random" endpoint with an enhanced version that includes:
   - Random error generation for testing purposes.
   - Random delay to simulate processing time.
   - Collection and exposure of system metrics.
   - Recording of price prediction histograms.

Key Components:
- ERROR_PROBABILITY: Probability of generating a random error.
- Custom Prometheus Gauges: CPU_USAGE, DISK_USAGE, MEMORY_USAGE, NETWORK_USAGE.
- Custom Prometheus Histogram: price_predictions.
"""

import random
import time
import numpy as np
from fastapi import HTTPException
from prometheus_client import Histogram, Gauge
import psutil
from services.app.app import app, gen_random_data
from services.app.app_stage_3 import instrumentator

# Define the probability of generating a random error
ERROR_PROBABILITY = 0.1

# Define Prometheus Gauges for system metrics
CPU_USAGE = Gauge('custom_cpu_usage_percent', 'CPU usage percent')
DISK_USAGE = Gauge('custom_disk_usage_percent', 'Disk usage percent')
MEMORY_USAGE = Gauge('custom_memory_usage_percent', 'Memory usage percent')
NETWORK_USAGE = Gauge('custom_network_usage_bytes_total', 'Network usage bytes')

# Define a Prometheus Histogram for price predictions
price_predictions = Histogram(
    "price_predictions",
    "Histogram of predictions",
    buckets=np.arange(0, 2e8 + 1, 2e7).tolist()
)

# Remove the original /random route if it exists
for route in app.routes:
    if route.path == "/random" and route.name == "get_random_prediction":
        app.routes.remove(route)

@app.get("/random")
def get_random_prediction() -> tuple:
    """
    Endpoint to get predictions based on random model parameters with added error simulation and metrics.

    Returns:
        tuple: A tuple containing the random model parameters and the prediction results.

    Raises:
        HTTPException: Randomly raises an HTTP 500 error for testing purposes.
    """
    if random.random() < ERROR_PROBABILITY:
        raise HTTPException(status_code=500, detail="Random failure for testing purposes")
    
    time.sleep(random.random())
    
    random_params = gen_random_data()
    predicted_price = app.handler.handle(random_params)['score']
    
    price_predictions.observe(predicted_price)

    # Update Prometheus Gauges with system metrics
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    DISK_USAGE.set(psutil.disk_usage('/').percent)
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    net_io = psutil.net_io_counters()
    NETWORK_USAGE.set(net_io.bytes_sent + net_io.bytes_recv)
    
    return (random_params, predicted_price)
