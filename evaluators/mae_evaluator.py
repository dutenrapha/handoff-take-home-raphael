import numpy as np
from evaluators.base_evaluator import BaseEvaluator

class MAE(BaseEvaluator):
    """Mean Absolute Error (MAE) evaluator."""

    def evaluate(self, ground_truth, predictions):
        """
        Compute MAE.

        Args:
            ground_truth (list or np.array): List of actual values.
            predictions (list or np.array): List of predicted values.

        Returns:
            float: MAE score (lower is better).
        """
        ground_truth = np.array(ground_truth)
        predictions = np.array(predictions)
        return np.mean(np.abs(ground_truth - predictions))
