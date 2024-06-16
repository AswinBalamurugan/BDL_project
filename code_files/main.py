from utils import *
from preprocess import *
from train_model import *
import psutil, argparse

# Paths to preprocessing & model training files
preprocess_path = os.path.join(os.path.join(cwd,"code_files"), "preprocess.py")
train_model_path = os.path.join(os.path.join(cwd,"code_files"), "train_model.py")

# Initialize Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total number of requests')
RESPONSE_TIME = Histogram('response_time', 'Response time in seconds')
PREDICTION_TIME = Histogram('prediction_time', 'Time taken to make a prediction')
DATA_PROCESSING_TIME = Histogram('data_processing_time', 'Time taken to preprocess the input data')
MEMORY_USAGE = Gauge('memory_usage', 'Memory usage of the application')
CPU_USAGE = Gauge('cpu_usage', 'CPU usage of the application')

# Create a FastAPI application
app = FastAPI()

# Route to handle prediction requests
@app.post("/predict")
async def predict(data: IrisData = Body(...)):
    global REQUEST_COUNT, RESPONSE_TIME, PREDICTION_TIME, DATA_PROCESSING_TIME, MEMORY_USAGE, CPU_USAGE
    REQUEST_COUNT.inc()
    with RESPONSE_TIME.time():
        with DATA_PROCESSING_TIME.time():
            spark = SparkSession.builder.appName("IrisClassification").getOrCreate()

            # Preprocess the input data
            processed_data = preprocess_single_data(spark, data)

        # Gauge CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        
        with PREDICTION_TIME.time():
            # Make predictions
            predictions = make_predictions(processed_data)

        # Release the session back to the pool
        spark.stop()

        MEMORY_USAGE.set(memory_info.total / 1024 / 1024 / 1024)  # in GB
        CPU_USAGE.set(cpu_percent)

        return predictions

# Start the Prometheus metrics server
@app.get('/metrics')
def get_metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help="Train model")
    args = parser.parse_args()

    if args.train:

        try:
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
        except OSError as error:
            pass

        # download the Iris dataset
        os.system(f"curl -o {iris_data_path} --url {iris_data_url}")

        # Preprocess the data
        preprocess_data()

        # Train the model
        train_model(train_path, test_path)

    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000
        )


'''

pkill -f 'spark'
pkill -f 'uvicorn'

'''