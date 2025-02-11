import pytest
import os
import json
import pandas as pd
from adapters.json_adapter import JSONAdapter

GROUND_TRUTH_JSON = "tests/data/test_ground_truth.json"
MISSING_JSON = "tests/data/non_existent.json"
MALFORMED_JSON = "tests/data/malformed.json"


VALID_JSON_DATA = {
    "rows": [
        {"sectionName": "Plumbing", "qty": 10, "rateUsd": 50, "rowTotalCostUsd": 500, "label": "Pipe", "uom": "EA", "category": "Water"},
        {"sectionName": "Electrical", "qty": 5, "rateUsd": 100, "rowTotalCostUsd": 500, "label": "Wiring", "uom": "LF", "category": "Power"}
    ]
}

@pytest.fixture
def create_test_files():
    """Fixture to create temporary JSON files for testing."""
    os.makedirs("tests/data", exist_ok=True)


    with open(GROUND_TRUTH_JSON, "w", encoding="utf-8") as file:
        json.dump(VALID_JSON_DATA, file)


    with open(MALFORMED_JSON, "w", encoding="utf-8") as file:
        file.write("{invalid_json}")

    yield 
  
    os.remove(GROUND_TRUTH_JSON)
    os.remove(MALFORMED_JSON)
    os.rmdir("tests/data")

def test_json_adapter_load_valid_json(create_test_files):
    """Test that JSONAdapter correctly loads a valid JSON file."""
    adapter = JSONAdapter(GROUND_TRUTH_JSON)
    df = adapter.to_dataframe()

    assert isinstance(df, pd.DataFrame), "Output should be a DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert list(df.columns) == ["sectionName", "qty", "rateUsd", "rowTotalCostUsd", "label", "uom", "category"], "Column names should match JSON keys"

def test_json_adapter_missing_file():
    """Test that JSONAdapter raises FileNotFoundError when the file does not exist."""
    with pytest.raises(FileNotFoundError):
        JSONAdapter(MISSING_JSON)

def test_json_adapter_malformed_json(create_test_files):
    """Test that JSONAdapter raises an error when the JSON is malformed."""
    with pytest.raises(json.JSONDecodeError):
        JSONAdapter(MALFORMED_JSON)

def test_json_adapter_missing_rows_key(create_test_files):
    """Test that JSONAdapter raises ValueError if the JSON is missing the 'rows' key."""
    missing_rows_json = "tests/data/missing_rows.json"

    with open(missing_rows_json, "w", encoding="utf-8") as file:
        json.dump({"test_n": 1}, file)

    with pytest.raises(ValueError):
        JSONAdapter(missing_rows_json).to_dataframe()

    os.remove(missing_rows_json)
