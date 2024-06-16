# Import necessary libraries
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import os, uvicorn
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler, StringIndexerModel, IndexToString
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
import mlflow.spark 
from fastapi import FastAPI, Body, Response
from pydantic import BaseModel
from datetime import datetime
from sklearn.metrics import f1_score

# download the Iris dataset
iris_data_url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
cwd = os.getcwd()
data_dir = os.path.join(cwd,"dataset")
model_dir = os.path.join(cwd,"model")
train_path = os.path.join(data_dir,"train_data")
test_path = os.path.join(data_dir,'test_data')
model_path = os.path.join(model_dir,"spark_rfc_model")
label_model_path = os.path.join(model_dir,"label_model")
pipe_model_path = os.path.join(model_dir,"pipe_model")
iris_data_path = os.path.join(data_dir, 'iris.csv')

# Define the data schema for prediction requests
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Function to preprocess a single data point
def preprocess_single_data(spark, data: IrisData):
    """
    This function pre-processes a single data point for the Iris classification task.

    Parameters:
    spark (SparkSession): The SparkSession object used to create DataFrames and DataSets.
    data (IrisData): An instance of the IrisData class containing the input data for preprocessing.

    Returns:
    DataFrame: A DataFrame containing the preprocessed data.

    The function first creates a feature vector from the input data using a VectorAssembler. 
    It then applies a StandardScaler to scale the features. 
    Finally, it returns the preprocessed DataFrame.
    """

    pipe_model = PipelineModel.load(pipe_model_path)
    
    data_list = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    data_df = spark.createDataFrame(data_list, ["sepal_length", "sepal_width", "petal_length", "petal_width"])
    
    # Preprocess data using the pipeline from preprocess_data function
    processed_data = pipe_model.transform(data_df)

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

    return pred