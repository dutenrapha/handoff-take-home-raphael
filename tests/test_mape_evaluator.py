import pytest
import numpy as np
from evaluators.mape_evaluator import MAPE

@pytest.fixture
def evaluator():
    return MAPE()

def test_mape_perfect_match(evaluator):
    """Test MAPE when predictions match ground truth exactly."""
    ground_truth = [100, 200, 300]
    predictions = [100, 200, 300]
    assert evaluator.evaluate(ground_truth, predictions) == 0.0

def test_mape_with_differences(evaluator):
    """Test MAPE with varied percentage errors."""
    ground_truth = [100, 200, 300]
    predictions = [110, 180, 330]
    expected_mape = (10/100 + 20/200 + 30/300) / 3 * 100
    assert np.isclose(evaluator.evaluate(ground_truth, predictions), expected_mape, atol=1e-2)

def test_mape_with_large_errors(evaluator):
    """Test MAPE with large percentage errors."""
    ground_truth = [100, 200, 300]
    predictions = [50, 400, 100]
    expected_mape = (50/100 + 200/200 + 200/300) / 3 * 100
    assert np.isclose(evaluator.evaluate(ground_truth, predictions), expected_mape, atol=1e-2)

def test_mape_with_zero_ground_truth(evaluator):
    """Test MAPE handling division by zero (should ignore and return mean)."""
    ground_truth = [0, 200, 300]
    predictions = [50, 190, 310]
    expected_mape = (10/200 + 10/300) / 2 * 100
    assert np.isclose(evaluator.evaluate(ground_truth, predictions), expected_mape, atol=1e-2)

def test_mape_with_empty_lists(evaluator):
    """Test MAPE with empty lists (should return NaN)."""
    ground_truth = []
    predictions = []
    assert np.isnan(evaluator.evaluate(ground_truth, predictions))
