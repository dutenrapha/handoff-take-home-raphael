from observers.base_observer import BaseObserver

class ConsoleLogger(BaseObserver):
    """Observer that logs model evaluation results to the console."""

    def update(self, model_name: str, metric_name: str, score: float):
        """
        Logs evaluation results to the console.

        Args:
            model_name (str): Name of the evaluated model.
            metric_name (str): Name of the evaluation metric.
            score (float): Computed score for the model.
        """
        print(f"[LOG] Model: {model_name} | Metric: {metric_name} | Score: {score:.4f}")
