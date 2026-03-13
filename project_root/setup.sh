#!/bin/bash
set -e

echo "Setting up Airflow Wine Clustering Lab"

# Remove existing .env file if it exists
rm -f .env
rm -rf ./logs ./plugins ./config

# Stop and remove containers
docker compose down -v

# Create Airflow directories
mkdir -p ./logs ./plugins ./config

# Write current user UID
echo "AIRFLOW_UID=$(id -u)" > .env

# Show Airflow config
docker compose run --rm airflow-cli airflow config list
