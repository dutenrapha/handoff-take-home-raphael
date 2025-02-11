import pytest
import pandas as pd
import os
from reports.csv_report import CSVReport

@pytest.fixture
def sample_results():
    """Fixture to create sample evaluation results."""
    return pd.DataFrame([
        {"model_name": "GPT-4", "MAE": 10.5, "MAPE": 12.3, "MRE": 0.15},
        {"model_name": "Llama-2", "MAE": 8.2, "MAPE": 10.8, "MRE": 0.12}
    ])

@pytest.fixture
def csv_report():
    """Fixture to create a CSVReport instance."""
    return CSVReport()

def test_csv_report_generation(csv_report, sample_results, tmp_path):
    """Test that CSVReport generates a valid CSV file."""
    output_file = tmp_path / "test_evaluation_report.csv"
    csv_report.generate_report(sample_results, str(output_file))

    assert output_file.exists(), "CSV report file should be created."

    # Read back the file to verify content
    df = pd.read_csv(output_file)

    assert list(df.columns) == ["model_name", "MAE", "MAPE", "MRE"], "CSV file should have correct columns."
    assert df.shape == (2, 4), "CSV file should contain 2 rows and 4 columns."
    assert df["model_name"].tolist() == ["GPT-4", "Llama-2"], "CSV should contain correct model names."
