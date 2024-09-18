FROM apache/airflow:2.6.3-python3.10

ENV AIRFLOW_HOME=/opt/airflow

USER root
RUN apt-get update && apt-get install -y wget curl unzip

USER airflow
COPY data_utils.py $AIRFLOW_HOME/dags/data_utils.py
COPY ./airflow/dags $AIRFLOW_HOME/dags
COPY ./airflow/setup_airflow.sh $AIRFLOW_HOME/setup_airflow.sh

RUN pip install -U pip
COPY requirements.txt $AIRFLOW_HOME/requirements.txt
RUN pip install --no-cache-dir -r $AIRFLOW_HOME/requirements.txt

WORKDIR $AIRFLOW_HOME

ENTRYPOINT ["/bin/sh", "-c", "$AIRFLOW_HOME/setup_airflow.sh"]
