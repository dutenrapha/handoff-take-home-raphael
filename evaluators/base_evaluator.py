from abc import ABC, abstractmethod
import numpy as np

class BaseEvaluator(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def evaluate(self, ground_truth, predictions):
        """
        Compute the evaluation metric.

        Args:
            ground_truth (list or np.array): List of actual values.
            predictions (list or np.array): List of predicted values.

        Returns:
            float: Computed metric score.
        """
        pass
