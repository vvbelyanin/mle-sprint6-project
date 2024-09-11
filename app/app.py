"""
services/app/app.py

This module defines a FastAPI application with three endpoints:

1. GET "/": Returns the status of the service.
2. POST "/predict": Accepts model parameters and returns predictions.
3. GET "/random": Generates random model parameters and returns predictions based on them.

The FastAPI application uses a custom handler, `FastApiHandler`, to process predictions.
"""

from fastapi import FastAPI
from services.app.fastapi_handler import FastApiHandler, gen_random_data

app = FastAPI()
app.handler = FastApiHandler()

@app.get("/")
def read_root() -> dict:
    """
    Root endpoint that returns the status of the service.

    Returns:
        dict: A dictionary indicating the service status.
    """
    return {"status": "Alive"}

@app.post("/predict")
def get_prediction_for_item(model_params: dict) -> dict:
    """
    Endpoint to get predictions based on the provided model parameters.

    Args:
        model_params (dict): A dictionary containing model parameters.

    Returns:
        dict: A dictionary containing the prediction results.
    """
    return app.handler.handle(model_params)

@app.get("/random")
def get_random_prediction() -> tuple:
    """
    Endpoint to get predictions based on random model parameters.

    Returns:
        tuple: A tuple containing the random model parameters and the prediction results.
    """
    random_params = gen_random_data()
    return (random_params, app.handler.handle(random_params))
