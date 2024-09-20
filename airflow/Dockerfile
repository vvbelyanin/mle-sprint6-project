# Use the official Apache Airflow base image with Python 3.10
FROM apache/airflow:2.6.3-python3.10

# Set the Airflow home directory
ENV AIRFLOW_HOME=/opt/airflow

# Set PYTHONPATH to include the Airflow home directory (so Python knows where to look for modules)
ENV PYTHONPATH=/opt/airflow

# Switch to root user to install system-level packages
USER root

# Update the package list and install essential tools like wget, curl, and unzip
RUN apt-get update && apt-get install -y wget curl unzip

# Switch back to the airflow user (non-root user) for security purposes
USER airflow

# Copy the `utils` folder into the Airflow home directory
COPY ./utils $AIRFLOW_HOME/utils

# Copy the `models` folder into the Airflow home directory
COPY ./models $AIRFLOW_HOME/models

# Copy the DAGs folder into the Airflow home directory
COPY ./airflow/dags $AIRFLOW_HOME/dags

# Copy the Airflow setup script into the Airflow home directory
COPY ./airflow/setup_airflow.sh $AIRFLOW_HOME/setup_airflow.sh

# Copy the Airflow configuration file into the Airflow home directory
COPY ./airflow/airflow.cfg $AIRFLOW_HOME/airflow.cfg

# Upgrade pip to the latest version
RUN pip install -U pip

# Copy the requirements.txt file into the Airflow home directory
COPY requirements.txt $AIRFLOW_HOME/requirements.txt

# Install Python dependencies listed in requirements.txt without caching
RUN pip install --no-cache-dir -r $AIRFLOW_HOME/requirements.txt

# Set the working directory to the Airflow home directory
WORKDIR $AIRFLOW_HOME

# Create a temporary folder inside the Airflow home directory for any temp files
RUN mkdir -p ./tmp

# Set the entry point for the container to execute the Airflow setup script
ENTRYPOINT ["/bin/sh", "-c", "$AIRFLOW_HOME/setup_airflow.sh"]
