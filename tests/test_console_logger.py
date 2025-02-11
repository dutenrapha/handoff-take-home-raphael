import pytest
from observers.console_logger import ConsoleLogger
from io import StringIO
import sys

@pytest.fixture
def console_logger():
    """Fixture to create an instance of ConsoleLogger."""
    return ConsoleLogger()

def test_console_logger_output(console_logger, capsys):
    """Test if ConsoleLogger correctly prints the expected log message."""
    console_logger.update("GPT-4", "MAE", 10.5)

    # Capture the printed output
    captured = capsys.readouterr()
    
    expected_output = "[LOG] Model: GPT-4 | Metric: MAE | Score: 10.5000\n"
    assert captured.out == expected_output, "ConsoleLogger output does not match expected format."
