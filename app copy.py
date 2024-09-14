import uvicorn
import os
import joblib
import json
import psutil
from fastapi import FastAPI, Body
import pandas as pd
from data_utils import get_mem_usage, new_columns, attrs
from data_utils import DataFrameProcessor, frequency_encoding, get_X_y
from data_utils import gen_random_data
from dotenv import load_dotenv
from typing import Dict
import time

from statsd import StatsClient

load_dotenv()

DATA_DIR = os.getenv('DATA_DIR')
MODEL_DIR = os.getenv('MODEL_DIR')
FITTED_MODEL = os.getenv('FITTED_MODEL')
MODEL_PARAMS = os.getenv('MODEL_PARAMS')


def load_row_from_json(json_data):
    row_loaded = pd.Series(json_data)

    row_loaded['fetch_date'] = pd.to_datetime(row_loaded['fetch_date'], format='%d-%m-%Y', errors='coerce')
    row_loaded['registration_date'] = pd.to_datetime(row_loaded['registration_date'], format='%d-%m-%Y', errors='coerce')
    row_loaded['last_date_as_primary'] = pd.to_datetime(row_loaded['last_date_as_primary'], format='%d-%m-%Y', errors='coerce')

    if pd.isna(row_loaded['income']):
        row_loaded['income'] = income_mean

    return pd.DataFrame([row_loaded])

def interpret_predictions(predictions, lang='rus'):
    if lang == 'rus':
        targets = target_names
    else:
        targets = target_names_eng
    res = {}
    for col, name in zip(predictions, targets):
        res[name] = int(col)
    return res

def refresh_metrics(st):
    global last_time, service_start
    st.incr('bank-rs.requests')
    
    if time.time() - last_time > time_delta:
        cpu_load = psutil.cpu_percent(interval=1)
        st.gauge('bank-rs.system.cpu_load', cpu_load)
        memory_info = psutil.virtual_memory().available / 1024 / 1024  # in MB
        st.gauge('bank-rs.system.memory_free_mb', memory_info)
        last_time = time.time()
    
    st.gauge('bank-rs.system.up_time', time.time() - service_start)
    st.incr("response_code.200")


count_requests = 0
service_start = time.time()

last_time = time.time()
time_delta = 10

model = joblib.load(MODEL_DIR + FITTED_MODEL)

with open(MODEL_DIR + MODEL_PARAMS, 'r') as f:
    json_data = json.load(f)
    target_names = json_data['target_names']
    target_names_eng = json_data['target_names_eng']
    income_mean = json_data['income_mean']

app = FastAPI(title="Bank RS")

stats_client = StatsClient(host="graphite", port=8125, prefix="bank-rs")

@app.get("/")
async def read_root() -> dict:
    refresh_metrics(stats_client)
    return {"status": "Alive"}

@app.get("/random")
async def get_random() -> dict:
    start_time = time.time()
    row = load_row_from_json(gen_random_data())
    predictions = model.predict(row)[0]

    response_time = time.time() - start_time
    stats_client.timing("response_time", response_time)

    predictions_dict = interpret_predictions(predictions, lang='eng')
    for target, prediction in predictions_dict.items():
        if prediction==1:
            stats_client.incr("target." + target)
    
    refresh_metrics(stats_client)
    return interpret_predictions(predictions)


@app.post("/predict")
async def predict(data: Dict = Body(...)): 
    start_time = time.time()

    row = load_row_from_json(data)
    predictions = model.predict(row)[0]
    response_time = time.time() - start_time

    stats_client.timing("response_time", response_time)

    predictions_dict = interpret_predictions(predictions, lang='eng')
    for target, prediction in predictions_dict.items():
        if prediction==1:
            stats_client.incr("target." + target)
    
    refresh_metrics(stats_client)
    return interpret_predictions(predictions)
