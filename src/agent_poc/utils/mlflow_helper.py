import mlflow
from dotenv import load_dotenv


def init(
    tracking_uri: str = "http://localhost:5001", experiment_name: str = "agent-poc"
):
    load_dotenv()  # Load environment variables from .env file
    mlflow.dspy.autolog()
    # Optional: Set a tracking URI and an experiment
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
