import os
from observers.base_observer import BaseObserver

class FileLogger(BaseObserver):
    """Observer that logs model evaluation results to a file."""

    def __init__(self, file_path: str):
        """
        Initializes the FileLogger.

        Args:
            file_path (str): Path to the log file.
        """
        self.file_path = file_path
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        """Ensures that the log directory exists."""
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def update(self, model_name: str, metric_name: str, score: float):
        """
        Logs evaluation results to a file.

        Args:
            model_name (str): Name of the evaluated model.
            metric_name (str): Name of the evaluation metric.
            score (float): Computed score for the model.
        """
        with open(self.file_path, "a", encoding="utf-8") as file:
            file.write(f"Model: {model_name} | Metric: {metric_name} | Score: {score:.4f}\n")
