# import necessary libraries
import prometheus_client
import os, uvicorn
import urllib.request
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler, StringIndexerModel, IndexToString
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
import mlflow.spark 
from fastapi import FastAPI, Body, Response
from pydantic import BaseModel
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from sklearn.metrics import f1_score

# create metrics for monitoring
REQUEST_COUNT = prometheus_client.Counter('request_count', 'Total number of requests')
RESPONSE_TIME = prometheus_client.Histogram('response_time', 'Response time in seconds')

# download the Iris dataset
iris_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cwd = os.getcwd()
data_dir = os.path.join(cwd,"dataset")
model_dir = os.path.join(cwd,"model")
train_path = os.path.join(data_dir,"train_data")
test_path = os.path.join(data_dir,'test_data')
model_path = os.path.join(model_dir,"spark_rfc_model")
label_model_path = os.path.join(model_dir,"label_model")
iris_data_path = os.path.join(data_dir, 'iris.csv')

try:
    os.mkdir(data_dir)
except OSError as error:
    pass

try:
    os.mkdir(model_dir)
except OSError as error:
    pass

# print(f"Downloading Iris dataset from {iris_data_url}")
# urllib.request.urlretrieve(iris_data_url, iris_data_path)
# print("Download complete.")

# create a SparkSession
spark = SparkSession.builder.appName("IrisClassification").getOrCreate()

process_pipe = []

# Create feature vector
vectorAssembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], outputCol="features")
process_pipe += [vectorAssembler]

ss= StandardScaler(inputCol="features",
                outputCol="feats",
                withMean= False,
                withStd=True)
process_pipe += [ss]

pipe = Pipeline(stages= process_pipe)

# define the data preprocessing function
def preprocess_data(ti):
    # read the Iris dataset
    iris_data = spark.read.csv(iris_data_path, header=False, inferSchema=True)
    iris_data = iris_data.toDF("sepal_length", "sepal_width", "petal_length", "petal_width", "label")

    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(iris_data)
    iris_data = labelIndexer.transform(iris_data)
    labelIndexer.save(label_model_path)

    iris_data = pipe.fit(iris_data).transform(iris_data)

    # Split data into training and testing sets
    train_data, test_data = iris_data.randomSplit([0.8, 0.2], seed=15)

    # Save training and testing data
    train_data.write.mode('overwrite').parquet(train_path)
    test_data.write.mode('overwrite').parquet(test_path)

    ti.xcom_push(key='train_path', value=train_path)
    ti.xcom_push(key='test_path', value=test_path)

    spark.stop()

# define the model training function
def train_model(ti):
    # Retrieve preprocessed data paths from XCom 
    train_path = ti.xcom_pull(task_ids='preprocess_data', key='train_path') 
    test_path = ti.xcom_pull(task_ids='preprocess_data', key='test_path')  

    # Create a new Spark session for model training
    spark = SparkSession.builder.appName("IrisClassification").getOrCreate()

    # Load preprocessed data
    train_data = spark.read.parquet(train_path)
    test_data = spark.read.parquet(test_path)

    # start an MLflow experiment
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("iris_classification")
    parent_exp_id = mlflow.get_experiment_by_name("iris_classification")
    
    # train the model
    with mlflow.start_run(run_name='Iris-model-build', experiment_id=parent_exp_id.experiment_id):
        # log model hyperparameters
        mlflow.log_param("num_trees", 100)
        mlflow.log_param("max_depth", 5)

        # define the machine learning pipeline
        rf = RandomForestClassifier(numTrees=100, maxDepth=5, featuresCol='feats',labelCol='indexedLabel', seed=15)

        model = rf.fit(train_data)

        # log the model
        mlflow.spark.log_model(model, "model")
        # Save the model to a specific path
        model.write().overwrite().save(model_path)

        # evaluate the model
        predictions = model.transform(test_data)

        metrics = predictions.select("prediction", "indexedLabel").toPandas()
        accuracy = f1_score(metrics['prediction'], metrics['indexedLabel'], average='weighted')

        mlflow.log_metric("accuracy", accuracy)

    # Stop the Spark session after training
    spark.stop()

# define the Airflow DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 5, 17),
    'retries': None,
    'retry_delay': timedelta(minutes=5),
}

with DAG('iris_classification', default_args=default_args, schedule_interval=None) as dag:
    # define the tasks
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    # set the task dependencies
    preprocess_task >> train_task

# Define the data schema for prediction requests
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Create a FastAPI application
app = FastAPI()

# Function to preprocess a single data point
def preprocess_single_data(data: IrisData):
    # Create a new Spark session for each request
    spark = SparkSession.builder.appName("IrisClassification").getOrCreate()
    
    data_list = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    data_df = spark.createDataFrame(data_list, ["sepal_length", "sepal_width", "petal_length", "petal_width"])
    
    # Preprocess data using the pipeline from preprocess_data function
    processed_data = pipe.fit(data_df).transform(data_df)
    
    return processed_data

# Function to make predictions using the loaded model
def make_predictions(processed_data):
    
    # Load the trained model
    model = RandomForestClassificationModel.load(model_path)

    # Make predictions
    predictions = model.transform(processed_data).withColumnRenamed("prediction", "indexedLabel")

    # Load the label indexer model
    labelmodel = StringIndexerModel.load(label_model_path)
    inverse_label = IndexToString(inputCol='indexedLabel',outputCol='label',labels=labelmodel.labels)
    pred = inverse_label.transform(predictions).toPandas()['label'].values[0]

    spark.stop()

    return pred

# Route to handle prediction requests
@app.post("/predict")
async def predict(data: IrisData = Body(...)):
    REQUEST_COUNT.inc()
    with RESPONSE_TIME.time():
        # Preprocess the input data
        processed_data = preprocess_single_data(data)

        # Make predictions
        predictions = make_predictions(processed_data)
        return predictions

# Start the Prometheus metrics server
@app.get('/metrics')
def get_metrics():
    return Response(
        content=prometheus_client.generate_latest(),
        media_type="text/plain"
    )

if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000
        )


'''
airflow users create \
    --username airflow \
    --firstname Airflow \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password airflow

    
pkill -f 'uvicorn'
'''