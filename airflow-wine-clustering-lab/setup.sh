#!/bin/bash
set -e

echo "Setting up Airflow Wine Clustering Lab"

rm -f .env
rm -rf ./logs ./plugins ./config

docker compose down -v

mkdir -p ./logs ./plugins ./config

echo "AIRFLOW_UID=$(id -u)" > .env

docker compose run --rm airflow-cli airflow config list
