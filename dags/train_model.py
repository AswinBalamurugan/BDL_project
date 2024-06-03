from utils import *

# define the model training function
def train_model(train_path, test_path):
    """
    This function trains a machine learning model using the preprocessed data.

    Parameters:
    ti (TaskInstance): The TaskInstance object from Airflow, used to pull data from XCom.

    Returns:
    None
    """

    # create a SparkSession
    spark = SparkSession.builder \
        .appName("IrisClassification") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

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
        mlflow.log_param("num_trees", 20)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("featuresCol", 'feats')
        mlflow.log_param("labelCol", 'indexedLabel')
        mlflow.log_param("seed", 15)

        # define the machine learning pipeline
        rf = RandomForestClassifier(numTrees=20, maxDepth=5, featuresCol='feats',labelCol='indexedLabel', seed=15)

        model = rf.fit(train_data)

        # Save the model to a specific path
        model.write().overwrite().save(model_path)

        # evaluate the model
        predictions = model.transform(test_data)

        metrics = predictions.select("prediction", "indexedLabel").toPandas()
        accuracy = f1_score(metrics['prediction'], metrics['indexedLabel'], average='weighted')

        mlflow.log_metric("F1 score", accuracy)

        spark.stop()

train_model(train_path, test_path)