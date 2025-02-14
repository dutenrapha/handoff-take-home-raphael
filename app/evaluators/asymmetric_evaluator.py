import numpy as np
from evaluators.base_evaluator import BaseEvaluator

class AsymmetricLoss(BaseEvaluator):
    """Asymmetric loss function evaluator to penalize underestimation more than overestimation."""

    def __init__(self, alpha=2.0):
        """
        Initialize the asymmetric loss evaluator.

        Args:
            alpha (float): Factor to penalize underestimation (default is 2.0).
                          If alpha > 1, underestimations are penalized more than overestimations.
        """
        self.alpha = alpha

    def evaluate(self, ground_truth, predictions):
        """
        Compute the asymmetric loss.

        Args:
            ground_truth (list or np.array): List of actual values.
            predictions (list or np.array): List of predicted values.

        Returns:
            float: Asymmetric loss score (lower is better).
        """
        ground_truth = np.array(ground_truth)
        predictions = np.array(predictions)

        error = predictions - ground_truth
        loss = np.where(error < 0, self.alpha * np.abs(error), np.abs(error))

        return np.mean(loss)
