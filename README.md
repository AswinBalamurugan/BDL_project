# MLOpsIris: A Harmonious Blend of ML, APIs, and Monitoring with Iris Dataset

This project aims to build a machine-learning model for classifying iris flower species using the Iris dataset. It includes a data preprocessing pipeline, model training, and a REST API for making predictions. The pipeline is orchestrated using Apache Airflow, and the REST API is built with FastAPI.

## Grafana Dashboard
![grafana dashboard](https://github.com/AswinBalamurugan/MLOps_Iris/blob/main/imgs/grafana.png)

## Prerequisites

- Python 3.7 or later
- Apache Spark
- MLflow
- FastAPI
- Prometheus
- Docker
- Grafana
- Apache Airflow

## Setup

1. Clone the repository:

```bash
git clone https://github.com/AswinBalamurugan/MLOps_Iris.git
cd iris-classification
```

2. Create a virtual environment and activate it.

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Applications
Airflow

- Initialize the Airflow database:

```bash
airflow db init
```

- Create an Airflow user:

```bash
airflow users create \
    --username airflow \
    --firstname Airflow \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password airflow
```

- Start the Airflow webserver and scheduler:

```bash
airflow standalone
```

Use `http://localhost:8080` to access the airflow webserver.

- Trigger the Airflow DAG to run the pipeline by openning `http://localhost:8080` in your web browser.

- Start the MLflow server:

```bash
mlflow ui
```
Open the MLflow UI at `http://localhost:5000` to view experiment runs and logged metrics.

## FastAPI

1. Build the Docker images:

```bash
docker-compose build
```

2. Start the applications using:

```bash
docker-compose up -d
```

This will start the following services:

- `iris-api`: The FastAPI application for making iris species predictions.
- `prometheus`: The Prometheus server for monitoring.
- `grafana`: The Grafana server for visualizing metrics.

## Accessing the Applications

- FastAPI application: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### Using the FastAPI application
Send a POST request to the `/predict` endpoint with the input data in JSON format:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' http://127.0.0.1:8000/predict
```

### Monitoring with Prometheus and Grafana
The FastAPI application exposes a `/metrics` endpoint that Prometheus can scrape to collect application metrics. 
The following metrics are exposed:

- `request_count`: Total number of requests
- `response_time`: Response time in seconds
- `prediction_time`: Time taken to make a prediction
- `data_processing_time`: Time taken to preprocess the input data
- `memory_usage`: Memory usage of the application (in GB)
- `cpu_usage`: CPU usage of the application (percentage)

### Grafana Dashboard setup
Once you access the Grafana application at http://localhost:3000, you need to add the prometheus(http://localhost:9090) as a source. 
The default login and password are **admin** and **admin**.

## Dockerfile
The `Dockerfile` installs the required dependencies and copies the application files to the Docker image. It exposes port *8000* for the FastAPI application and sets the entrypoint to run the `main.py` file.
### Docker Compose
The `docker-compose.yml` file defines the following services:

- `iris-api`: Builds the FastAPI application image from the current directory and exposes port 8000. It also mounts the `dataset` directory from the host machine.
- `prometheus`: Runs the Prometheus server and mounts the `prometheus.yml` configuration file.
- `grafana`: Runs the Grafana server, exposes port *3000*, and mounts persistent storage volumes for Grafana data and dashboards (if present).

The `volumes` section defines persistent storage volumes for Airflow DAGs, logs, and Grafana data.

## Stopping the Applications
To stop the applications, run the following command:
```bash 
docker-compose down 
```
This will stop and remove the containers, networks, and volumes created by Docker Compose.


## To kill the processes in ports use the below commands

```bash
pkill -f "airflow"
pkill -f "python"
```
Make sure to adjust the file paths and configurations according to your environment.
