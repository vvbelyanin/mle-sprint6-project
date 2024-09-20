FROM python:3.10-slim

# Set the working directory
WORKDIR /

# Copy requirements and install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application files and the model
COPY app.py /app.py
COPY utils /utils
COPY models /models

# Expose the port
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
