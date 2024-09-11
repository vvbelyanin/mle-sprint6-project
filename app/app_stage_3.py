"""
services/app/app_stage_3.py

This module enhances the FastAPI application with Prometheus instrumentation for monitoring.

1. Imports the FastAPI application from `services.app.app`.
2. Uses `PrometheusFastAPIInstrumentator` to instrument and expose metrics for the FastAPI application.

The metrics can be used to monitor the performance and usage of the application.
"""

from prometheus_fastapi_instrumentator import Instrumentator
from services.app.app import app

# Create an instance of the Instrumentator
instrumentator = Instrumentator()

# Instrument the FastAPI application and expose the metrics endpoint
instrumentator.instrument(app).expose(app)
