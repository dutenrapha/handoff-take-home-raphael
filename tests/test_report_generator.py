import pytest
import pandas as pd
import os
from reports.report_generator import ReportGenerator

@pytest.fixture
def sample_results():
    """Fixture to create sample evaluation results."""
    return pd.DataFrame([
        {"model_name": "GPT-4", "MAE": 10.5, "MAPE": 12.3, "MRE": 0.15},
        {"model_name": "Llama-2", "MAE": 8.2, "MAPE": 10.8, "MRE": 0.12}
    ])

def test_csv_report_generation_with_report_generator(sample_results, tmp_path):
    """Test that ReportGenerator correctly generates a CSV report."""
    output_file = tmp_path / "test_evaluation_report.csv"
    report_generator = ReportGenerator(format="csv")
    report_generator.generate(sample_results, str(output_file))

    assert output_file.exists(), "CSV report file should be created."

def test_json_report_generation_with_report_generator(sample_results, tmp_path):
    """Test that ReportGenerator correctly generates a JSON report."""
    output_file = tmp_path / "test_evaluation_report.json"
    report_generator = ReportGenerator(format="json")
    report_generator.generate(sample_results, str(output_file))

    assert output_file.exists(), "JSON report file should be created."

def test_invalid_report_format():
    """Test that ReportGenerator raises an error for unsupported formats."""
    with pytest.raises(ValueError, match="Unsupported report format: xml"):
        ReportGenerator(format="xml")
