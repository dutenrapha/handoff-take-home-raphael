import numpy as np
from evaluators.base_evaluator import BaseEvaluator

class MRE(BaseEvaluator):
    """Mean Relative Error (MRE) evaluator."""

    def evaluate(self, ground_truth, predictions):
        """
        Compute MRE.

        Args:
            ground_truth (list or np.array): List of actual values.
            predictions (list or np.array): List of predicted values.

        Returns:
            float: MRE score (lower is better).
        """
        ground_truth = np.array(ground_truth)
        predictions = np.array(predictions)

        ground_truth = np.where(ground_truth == 0, np.nan, ground_truth)

        return np.nanmean(np.abs((ground_truth - predictions) / ground_truth))
