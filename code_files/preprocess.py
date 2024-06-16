from utils import *

# define the data preprocessing function
def preprocess_data():
    """
    This function pre-processes the Iris dataset by performing feature engineering, scaling, and splitting it into training and testing sets.

    Parameters:
    ti (TaskInstance): The TaskInstance object from Airflow, used to push data to XCom.

    Returns:
    None
    """

    # create a SparkSession
    spark = (SparkSession.builder.appName("IrisClassification").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

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

    # read the Iris dataset
    iris_data = spark.read.csv(iris_data_path, header=True, inferSchema=True)

    labelIndexer = StringIndexer(inputCol="species", outputCol="indexedLabel").fit(iris_data)
    iris_data = labelIndexer.transform(iris_data)
    labelIndexer.write().overwrite().save(label_model_path)

    pipe_model = pipe.fit(iris_data)
    iris_data = pipe_model.transform(iris_data)

    pipe_model.write().overwrite().save(pipe_model_path)

    # Stratified sampling with proportional representation
    train_data = iris_data.sampleBy("indexedLabel", fractions={0: 0.8, 1: 0.8, 2: 0.8}, seed=15)
    test_data = iris_data.subtract(train_data)

    # Save training and testing data
    train_data.write.mode('overwrite').parquet(train_path)
    test_data.write.mode('overwrite').parquet(test_path)

    spark.stop()
