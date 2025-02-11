import pytest
import os
from observers.file_logger import FileLogger

LOG_FILE = "tests/logs/test_evaluation_log.txt"

@pytest.fixture
def file_logger():
    """Fixture to create a FileLogger instance with a test log file."""
    os.makedirs("tests/logs", exist_ok=True)  # Ensure test directory exists
    return FileLogger(LOG_FILE)

def test_file_logger_output(file_logger):
    """Test if FileLogger correctly writes log messages to a file."""
    file_logger.update("GPT-4", "MAE", 10.5)
    file_logger.update("Llama-2", "MAPE", 12.34)

    # Read log file contents
    with open(LOG_FILE, "r", encoding="utf-8") as file:
        lines = file.readlines()

    assert len(lines) == 2, "Log file should contain two entries."
    assert lines[0].strip() == "Model: GPT-4 | Metric: MAE | Score: 10.5000"
    assert lines[1].strip() == "Model: Llama-2 | Metric: MAPE | Score: 12.3400"

def test_file_logger_creates_file(file_logger):
    """Test if FileLogger creates the log file when it doesn't exist."""
    assert os.path.exists(LOG_FILE), "Log file should be created by FileLogger."

def teardown_module(module):
    """Cleanup: Remove the test log file after tests."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    os.rmdir("tests/logs")
