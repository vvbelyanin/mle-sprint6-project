"""
services/app/fastapi_handler.py

This module provides a handler for processing model predictions in a FastAPI application.

1. Defines the `FastApiHandler` class to handle predictions and parameter validation.
2. Loads a pre-trained model from a pickle file.
3. Provides functions to generate sample data and random data for testing purposes.

Key Components:
- REQUIRED_PARAMS: List of required model parameters.
- MODEL_PATH: Path to the pre-trained model file.
- FastApiHandler: Class to handle model predictions and parameter validation.
- sample_data: Function to generate a sample set of model parameters.
- gen_random_data: Function to generate a random set of model parameters.
"""

from random import randint, uniform
import pickle
from pprint import pprint
import pandas as pd
from geopy.distance import geodesic
import os

# List of required model parameters
REQUIRED_PARAMS = [
    'floor', 'is_apartment', 'kitchen_area', 'living_area', 'rooms',
    'total_area', 'building_id', 'build_year', 'building_type_int', 
    'latitude', 'longitude', 'ceiling_height', 'flats_count', 'floors_total', 
    'has_elevator'
]

# Path to the pre-trained model file
MODEL_PATH = 'services/models/loaded_model.pkl'

class FastApiHandler:
    """
    A handler class for managing model predictions and parameter validation.
    """
    def __init__(self):
        """Initializes the handler and loads the model."""
        self.required_model_params = REQUIRED_PARAMS
        self.load_model(model_path=MODEL_PATH)

    def load_model(self, model_path: str):
        """
        Loads a pre-trained model from a pickle file.

        Args:
            model_path (str): Path to the model pickle file.

        Raises:
            Various exceptions if loading the model fails.
        """
        try:
            with open(model_path, 'rb') as model_file:
                self.model = pickle.load(model_file)
        except pickle.UnpicklingError:
            print("Error unpickling the data. The file may be corrupted or not a valid pickle file.")
        except EOFError:
            print("Reached end of file unexpectedly. The file may be corrupted.")
        except ImportError:
            print("Required module for unpickling not found.")
        except AttributeError as e:
            print(f"An attribute referenced during unpickling does not exist: {e}")
        except Exception as e:
            print(f"An unexpected error occurred, failed to load model: {e}")

    def price_predict(self, model_params: dict) -> float:
        """
        Predicts the price based on the provided model parameters.

        Args:
            model_params (dict): Dictionary of model parameters.

        Returns:
            float: Predicted price.
        """
        df_sample = pd.DataFrame(model_params, index=[0])
        return self.model.predict(df_sample)[0]

    def validate_params(self, model_params: dict) -> bool:
        """
        Validates that the provided parameters match the required model parameters.

        Args:
            model_params (dict): Dictionary of model parameters.

        Returns:
            bool: True if validation is successful, False otherwise.
        """
        if set(model_params.keys()) == set(self.required_model_params):
            print("All model params exist")
        else:
            print("Not all model params exist")
            return False
        return True

    def handle(self, model_params: dict) -> dict:
        """
        Handles the prediction request by validating parameters and predicting the price.

        Args:
            model_params (dict): Dictionary of model parameters.

        Returns:
            dict: Dictionary containing the prediction result or an error message.
        """
        if not self.validate_params(model_params):
            return {"Error": "Problem with parameters"}

        print("Predicting for model_params:")
        pprint(model_params, sort_dicts=False)
        try:
            predicted_price = self.price_predict(model_params)
            return {"score": predicted_price}
        except ValueError as ve:
            print(f"Value Error: {ve}")
            return {"Error": "Invalid input data format or value"}
        except TypeError as te:
            print(f"Type Error: {te}")
            return {"Error": "Incorrect input data type"}
        except AttributeError as ae:
            print(f"Attribute Error: {ae}")
            return {"Error": "Model or pipeline attribute issue"}
        except IndexError as ie:
            print(f"Index Error: {ie}")
            return {"Error": "Indexing issue with input data"}
        except KeyError as ke:
            print(f"Key Error: {ke}")
            return {"Error": "Missing key in input data"}
        except Exception as e:
            print(f"Error while handling request: {e}")
            return {"Error": "Problem with request"}

def sample_data() -> dict:
    """
    Generates a sample set of model parameters.

    Returns:
        dict: A dictionary containing sample model parameters.
    """
    return {
        "floor": 1, 
        "is_apartment": 0, 
        "kitchen_area": 7.0, 
        "living_area": 27.0, 
        "rooms": 2, 
        "total_area": 40.0, 
        "building_id": 764, 
        "build_year": 1936, 
        "building_type_int": 1, 
        "latitude": 55.74044418334961, 
        "longitude": 37.52492141723633, 
        "ceiling_height": 3.0, 
        "flats_count": 63, 
        "floors_total": 7, 
        "has_elevator": 1
    }

def gen_random_data() -> dict:
    """
    Generates a random set of model parameters.

    Returns:
        dict: A dictionary containing random model parameters.
    """
    return {
        "floor": randint(1, 100), 
        "is_apartment": randint(0, 1), 
        "kitchen_area": uniform(1, 100), 
        "living_area": uniform(1, 200), 
        "rooms": randint(1, 10), 
        "total_area": uniform(1, 300), 
        "building_id": randint(1, 20000), 
        "build_year": randint(1900, 2030),  
        "building_type_int": randint(1, 10), 
        "latitude": uniform(54, 56), 
        "longitude": uniform(36, 38), 
        "ceiling_height": uniform(1, 5), 
        "flats_count": randint(1, 1000), 
        "floors_total": randint(1, 100), 
        "has_elevator": randint(0, 1)
    }

if __name__ == "__main__":
    # Create a test request
    test_params = sample_data()

    # Create a request handler for the API
    handler = FastApiHandler()

    # Make a test request
    response = handler.handle(test_params)
    print(f"Response: {response}")
