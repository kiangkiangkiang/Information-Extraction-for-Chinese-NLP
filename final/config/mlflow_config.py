# Add MLflow for experiment # change mlflow to False
MLFLOW = False
is_experiment = True
os.environ["MLFLOW_TRACKING_URI"] = "http://ec2-44-213-176-187.compute-1.amazonaws.com:7003"
os.environ["MLFLOW_TRACKING_USERNAME"] = "luka"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "luka"
