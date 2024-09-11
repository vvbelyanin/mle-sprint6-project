
import uvicorn
import os
import joblib
import json

from fastapi import FastAPI, Request
import pandas as pd

from data_utils import get_mem_usage, new_columns, attrs
from data_utils import upload_to_s3, download_from_s3, show_s3_folder
from data_utils import DataFrameProcessor, frequency_encoding, get_X_y

from dotenv import load_dotenv
import numpy as np
from fastapi import FastAPI, Body
from typing import Dict

# Load environment variables from the .env file
load_dotenv()

env_var_names = [
    'UVICORN_HOST', 'UVICORN_PORT', 'BASE_URL',
    'LOCAL_MODEL_PATH', 'FITTED_MODEL', 'MODEL_PARAMS'
]

for var in env_var_names:
    globals()[var] = os.getenv(var)


model = joblib.load(LOCAL_MODEL_PATH + FITTED_MODEL)

with open(LOCAL_MODEL_PATH + MODEL_PARAMS, 'r') as f:
    json_data = json.load(f)
    target_names = json_data['target_names']
    income_mean = json_data['income_mean']

def load_row_from_json_old(json_file):
    row_loaded = pd.read_json(json_file, typ='series')
    

    row_loaded['fetch_date'] = pd.to_datetime(row_loaded['fetch_date'], format='%d-%m-%Y')
    row_loaded['registration_date'] = pd.to_datetime(row_loaded['registration_date'], format='%d-%m-%Y')
    row_loaded['last_date_as_primary'] = pd.to_datetime(row_loaded['last_date_as_primary'], format='%d-%m-%Y')

    if row_loaded['income'] is None:
        row_loaded['income'] = income_mean

    return pd.DataFrame([row_loaded])


def load_row_from_json(json_data):
    # Convert the JSON dictionary into a Pandas Series
    row_loaded = pd.Series(json_data)

    # Convert specific fields to datetime
    row_loaded['fetch_date'] = pd.to_datetime(row_loaded['fetch_date'], format='%d-%m-%Y', errors='coerce')
    row_loaded['registration_date'] = pd.to_datetime(row_loaded['registration_date'], format='%d-%m-%Y', errors='coerce')
    row_loaded['last_date_as_primary'] = pd.to_datetime(row_loaded['last_date_as_primary'], format='%d-%m-%Y', errors='coerce')

    # Handle missing 'income' field by filling with the mean value if missing
    if pd.isna(row_loaded['income']):
        row_loaded['income'] = income_mean

    # Return a DataFrame with one row for further processing
    return pd.DataFrame([row_loaded])

def interpret_predictions(predictions):
    res = {}
    for col, name in zip(predictions, target_names):
        res[name] = int(col)
    return res

app = FastAPI(title="Bank RS")

@app.get("/")
async def read_root() -> dict:
    return {"status": "Alive"}

@app.post("/predict")
async def predict(data: Dict = Body(...)): 
    row = load_row_from_json(data)
    prediction = model.predict(row)[0]
    return interpret_predictions(prediction)


def main():
    try:
        uvicorn.run(
            app,
            host=UVICORN_HOST,
            port=int(UVICORN_PORT),
            log_level="info"
        )
    except KeyboardInterrupt:
        print("Process terminated by user (Ctrl+C).")


if __name__ == "__main__":
    main()


