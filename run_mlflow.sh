#!/bin/bash
# /run_mlflow.sh
# Load environment variables from the .env file and export them to the shell
export $(cat .env | xargs)

# Construct the PostgreSQL connection string from the environment variables and export it
export POSTGRES_SQL_CONN=postgresql://$DB_DESTINATION_USER:$DB_DESTINATION_PASSWORD@\
$DB_DESTINATION_HOST:$DB_DESTINATION_PORT/$DB_DESTINATION_NAME

# Start the MLflow server
mlflow server \
  --backend-store-uri $POSTGRES_SQL_CONN \
  --registry-store-uri $POSTGRES_SQL_CONN \
  --default-artifact-root s3://$S3_BUCKET_NAME \
  --no-serve-artifacts
