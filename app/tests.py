"""
services/app/tests.py

This module provides unit tests for the FastAPI application using the unittest framework.

1. Tests the root endpoint for status check.
2. Tests the prediction endpoint with empty data, random data, and various error scenarios.
3. Tests the prediction endpoint with valid data.

Key Components:
- TestOnline: Test case class for the FastAPI application.
"""

import unittest
from fastapi.testclient import TestClient
from services.app.app import app
from services.app.fastapi_handler import sample_data

class TestOnline(unittest.TestCase):
    """
    Unit test class for testing the FastAPI application.
    """
    def setUp(self):
        """Sets up test data for the unit tests."""
        self.test_data = sample_data()

    def test_root(self):
        """Tests the root endpoint for service status."""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "Alive"})

    def test_predict_empty_data(self):
        """Tests the prediction endpoint with empty data."""
        response = client.get("/predict")
        self.assertEqual(response.status_code, 405)

    def test_predict_random_data(self):
        """Tests the prediction endpoint with random data."""
        response = client.get("/random")
        self.assertIsInstance(response.json()[1]['score'], float)

    def test_predict_error_data(self):
        """Tests the prediction endpoint with various error scenarios."""
        # Empty json
        response = client.post("/predict", json={})
        self.assertEqual(response.json(), {"Error": "Problem with parameters"})

        # Lack of params
        error_data = self.test_data.copy()
        del error_data['floor']
        response = client.post("/predict", json=error_data)
        self.assertEqual(response.json(), {'Error': 'Problem with parameters'})

        # Excess of params
        error_data = self.test_data.copy()
        error_data["extra_data"] = 42
        response = client.post("/predict", json=error_data)
        self.assertEqual(response.json(), {'Error': 'Problem with parameters'})

        # Wrong params name
        error_data = self.test_data.copy()
        del error_data['floor']
        error_data['floor_other'] = 1
        response = client.post("/predict", json=error_data)
        self.assertEqual(response.json(), {'Error': 'Problem with parameters'})

        # Wrong data format
        error_data = self.test_data.copy()
        error_data['floor'] = 'one'
        response = client.post("/predict", json=error_data)
        self.assertEqual(response.json(), {'Error': 'Invalid input data format or value'})

    def test_predict_ok(self):
        """Tests the prediction endpoint with valid data."""
        with TestClient(app) as client:
            response = client.post("/predict", json=self.test_data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {'score': 16694033.207565399})

if __name__ == '__main__':
    client = TestClient(app)
    unittest.main()
