import pytest
import numpy as np
from evaluators.mre_evaluator import MRE

@pytest.fixture
def evaluator():
    return MRE()

def test_mre_perfect_match(evaluator):
    """Test MRE when predictions match ground truth exactly."""
    ground_truth = [100, 200, 300]
    predictions = [100, 200, 300]
    assert evaluator.evaluate(ground_truth, predictions) == 0.0

def test_mre_with_differences(evaluator):
    """Test MRE with varied relative errors."""
    ground_truth = [100, 200, 300]
    predictions = [110, 180, 330]
    expected_mre = (10/100 + 20/200 + 30/300) / 3
    assert np.isclose(evaluator.evaluate(ground_truth, predictions), expected_mre, atol=1e-6)

def test_mre_with_large_errors(evaluator):
    """Test MRE with significant prediction errors."""
    ground_truth = [100, 200, 300]
    predictions = [50, 400, 100]
    expected_mre = (50/100 + 200/200 + 200/300) / 3
    assert np.isclose(evaluator.evaluate(ground_truth, predictions), expected_mre, atol=1e-6)

def test_mre_with_zero_ground_truth(evaluator):
    """Test MRE handling division by zero (should ignore and return mean)."""
    ground_truth = [0, 200, 300]
    predictions = [50, 190, 310]
    expected_mre = (10/200 + 10/300) / 2
    assert np.isclose(evaluator.evaluate(ground_truth, predictions), expected_mre, atol=1e-6)

def test_mre_with_empty_lists(evaluator):
    """Test MRE with empty lists (should return NaN)."""
    ground_truth = []
    predictions = []
    assert np.isnan(evaluator.evaluate(ground_truth, predictions))
