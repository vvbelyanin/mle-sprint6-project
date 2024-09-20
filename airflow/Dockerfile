# airflow/Dockerfile
#
# Use the official Apache Airflow base image with Python 3.10
FROM apache/airflow:2.6.3-python3.10

# Set the Airflow home directory & PYTHONPATH
ENV AIRFLOW_HOME=/opt/airflow
ENV PYTHONPATH=/opt/airflow

# Update the package list and install essential tools like wget, curl, and unzip
USER root
RUN apt-get update && apt-get install -y wget curl unzip

# Copy the needed folders/files into the Airflow home directory
USER airflow
COPY ./utils $AIRFLOW_HOME/utils
COPY ./models $AIRFLOW_HOME/models
COPY ./airflow/dags $AIRFLOW_HOME/dags
COPY ./airflow/setup_airflow.sh $AIRFLOW_HOME/setup_airflow.sh
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
