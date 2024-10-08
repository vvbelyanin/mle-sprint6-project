# /run_mimic_load.py
"""
This script performs load testing on a FastAPI application by sending a series of requests
to specific endpoints.

1. Sends GET requests to the "/random" endpoint.
2. Sends GET requests to the "/" endpoint.
3. Introduces random delays between requests to simulate real-world usage patterns.

Key Components:
- NUM_REQUESTS: The total number of requests to be sent.
"""

import time
import random
import requests

# Total number of requests to be sent during the load test
NUM_REQUESTS = 10000

for i in range(NUM_REQUESTS):
    """
    Send requests to the specified endpoints and introduce random delays.

    Iterates NUM_REQUESTS times, sending GET requests to the "/random" and "/metrics"
    endpoints of the FastAPI application running on localhost, and sleeps for a random
    duration between requests to simulate load.
    """
    print(f'\rRequest: {i+1} of {NUM_REQUESTS}', end='', flush=True)
    requests.get('http://localhost:8000/random', timeout=(5, 5))
    time.sleep(random.random()*0.1)
    requests.get('http://localhost:8000/', timeout=(5, 5))
    time.sleep(random.random()*0.1)
