from abc import ABC, abstractmethod

class BaseObserver(ABC):
    """Abstract base class for observers that listen to model evaluation events."""

    @abstractmethod
    def update(self, model_name: str, metric_name: str, score: float):
        """
        Handles the event when a model is evaluated.

        Args:
            model_name (str): Name of the evaluated model.
            metric_name (str): Name of the evaluation metric.
            score (float): Computed score for the model.
        """
        pass
