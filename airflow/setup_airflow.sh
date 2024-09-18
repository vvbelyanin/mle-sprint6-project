#!/bin/bash

# Initialize the Airflow metadata database
airflow db migrate

# Create admin user
airflow users delete --username admin

airflow users create \
    --username admin \
    --firstname Admin \
    --lastname Admin \
    --role Admin \
    --email admin@example.com \
    --password admin

# Add Yandex Cloud connection
airflow connections delete yandexcloud_default
airflow connections add yandexcloud_default \
    --conn-type aws \
    --conn-login "$AWS_ACCESS_KEY_ID" \
    --conn-password "$AWS_SECRET_ACCESS_KEY" \
    --conn-extra "{\"region_name\":\"$AWS_REGION\", \"endpoint_url\":\"$S3_ENDPOINT_URL\"}"

# Start Airflow standalone
airflow standalone
