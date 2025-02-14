import pytest
import numpy as np
from evaluators.mae_evaluator import MAE

@pytest.fixture
def evaluator():
    return MAE()

def test_mae_perfect_match(evaluator):
    """Test MAE when predictions are identical to ground truth."""
    ground_truth = [100, 200, 300]
    predictions = [100, 200, 300]
    assert evaluator.evaluate(ground_truth, predictions) == 0.0

def test_mae_with_differences(evaluator):
    """Test MAE with varied differences."""
    ground_truth = [100, 200, 300]
    predictions = [110, 190, 290]
    assert np.isclose(evaluator.evaluate(ground_truth, predictions), 10.0, atol=1e-6)

def test_mae_with_large_errors(evaluator):
    """Test MAE with significant prediction errors."""
    ground_truth = [100, 200, 300]
    predictions = [50, 400, 100]
    assert np.isclose(evaluator.evaluate(ground_truth, predictions), 150.0, atol=1e-6)

def test_mae_with_empty_lists(evaluator):
    """Test MAE with empty input lists (should return NaN)."""
    ground_truth = []
    predictions = []
    assert np.isnan(evaluator.evaluate(ground_truth, predictions))
