#!/bin/bash
# /run_mlflow.sh
# Load environment variables from the .env file and export them to the shell
export $(cat .env | xargs)

# Construct the PostgreSQL connection string from the environment variables and export it
export POSTGRES_SQL_CONN=postgresql://$DB_DESTINATION_USER:$DB_DESTINATION_PASSWORD@\
$DB_DESTINATION_HOST:$DB_DESTINATION_PORT/$DB_DESTINATION_NAME

# Start the MLflow server
mlflow server \
  --backend-store-uri $POSTGRES_SQL_CONN \   # Use the PostgreSQL backend store URI for experiment tracking
  --registry-store-uri $POSTGRES_SQL_CONN \  # Use the same PostgreSQL backend for the model registry
  --default-artifact-root s3://$S3_BUCKET_NAME \  # Specify the default root location in S3 for storing artifacts
  --no-serve-artifacts   # Disable MLflow's artifact serving feature (since artifacts are stored directly in S3)
