# Iris Classification Project

This project aims to build a machine learning model for classifying iris flower species using the Iris dataset. It includes a data preprocessing pipeline, model training, and a REST API for making predictions. The pipeline is orchestrated using Apache Airflow, and the REST API is built with FastAPI.

## Prerequisites

- Python 3.7 or later
- Apache Spark
- MLflow
- FastAPI
- Prometheus
- Apache Airflow

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/iris-classification.git
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
airflow webserver -D  # Open http://localhost:8080 in your web browser
airflow scheduler -D
```

- Trigger the Airflow DAG to run the pipeline by openning `http://localhost:8080` in your web browser.

- Start the MLflow server:

```bash
mlflow UI
```
Open the MLflow UI at `http://localhost:5000` to view experiment runs and logged metrics.

## FastAPI

Start the FastAPI server:

```bash
uvicorn dags.main:app --reload
```

Send a POST request to the /predict endpoint with the input data in JSON format:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' http://127.0.0.1:8000/predict
```

Or access the API @ `http://127.0.0.1:8000/docs` to use it.

## Project Structure 

- `dags/main.py`: FastAPI application with the REST API endpoint
- `dags/main.py`: Airflow DAG for orchestrating the pipeline
- `data/`: Directory for storing the Iris dataset and preprocessed data
- `model/`: Directory for storing the trained models
- `mlruns/`: Directory for MLflow experiment tracking
- `requirements.txt`: File listing the Python dependencies

## Explanation for main.py
This code defines an Airflow DAG with two tasks: `preprocess_data` and `train_model`. The `preprocess_data` task reads the Iris dataset, performs data preprocessing steps (label indexing and feature vector assembly), splits the data into training and testing sets, and saves them as Parquet files. The train_model task retrieves the preprocessed data paths from XCom, loads the data, trains a Random Forest Classifier model, logs the model and metrics to MLflow, and evaluates the model's performance using the F1 score.


## To kill the processes in ports use the below commands

```bash
pkill -f "airflow webserver"
pkill -f "airflow scheduler"
pkill -f "uvicorn"
```
Make sure to adjust the file paths and configurations according to your environment.
