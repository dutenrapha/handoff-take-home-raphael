import pytest
import pandas as pd
import os
import json
from reports.json_report import JSONReport

@pytest.fixture
def sample_results():
    """Fixture to create sample evaluation results."""
    return pd.DataFrame([
        {"model_name": "GPT-4", "MAE": 10.5, "MAPE": 12.3, "MRE": 0.15},
        {"model_name": "Llama-2", "MAE": 8.2, "MAPE": 10.8, "MRE": 0.12}
    ])

@pytest.fixture
def json_report():
    """Fixture to create a JSONReport instance."""
    return JSONReport()

def test_json_report_generation(json_report, sample_results, tmp_path):
    """Test that JSONReport generates a valid JSON file."""
    output_file = tmp_path / "test_evaluation_report.json"
    json_report.generate_report(sample_results, str(output_file))

    assert output_file.exists(), "JSON report file should be created."

    # Read back the file to verify content
    with open(output_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert isinstance(data, list), "JSON report should be a list of records."
    assert len(data) == 2, "JSON file should contain two records."
    assert data[0]["model_name"] == "GPT-4", "First model name should be 'GPT-4'."
    assert data[1]["MAE"] == 8.2, "Second model's MAE should be 8.2."
