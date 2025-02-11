import pytest
from observers.evaluation_notifier import EvaluationNotifier
from observers.console_logger import ConsoleLogger
from observers.file_logger import FileLogger
import os

@pytest.fixture
def notifier():
    """Fixture to create an EvaluationNotifier instance."""
    return EvaluationNotifier()

@pytest.fixture
def console_logger():
    """Fixture to create a ConsoleLogger instance."""
    return ConsoleLogger()

@pytest.fixture
def file_logger(tmp_path):
    """Fixture to create a FileLogger instance with a fresh temporary log file."""
    log_file = tmp_path / "test_evaluation_log.txt"
    return FileLogger(str(log_file))

def test_notifier_with_console_logger(notifier, console_logger, capsys):
    """Test if EvaluationNotifier correctly notifies the ConsoleLogger."""
    notifier.add_observer(console_logger)
    notifier.notify("GPT-4", "MAE", 10.5)

    captured = capsys.readouterr()
    
    expected_output = "[LOG] Model: GPT-4 | Metric: MAE | Score: 10.5000\n"
    assert captured.out == expected_output, "EvaluationNotifier did not notify ConsoleLogger correctly."

def test_notifier_with_file_logger(notifier, file_logger):
    """Test if EvaluationNotifier correctly notifies the FileLogger."""
    notifier.add_observer(file_logger)
    notifier.notify("GPT-4", "MAE", 10.5)

    with open(file_logger.file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    assert len(lines) == 1, "Log file should contain one entry."
    assert lines[0].strip() == "Model: GPT-4 | Metric: MAE | Score: 10.5000"

def test_notifier_with_multiple_observers(notifier, console_logger, file_logger, capsys):
    """Test if EvaluationNotifier correctly notifies multiple observers."""
    notifier.add_observer(console_logger)
    notifier.add_observer(file_logger)

    notifier.notify("Llama-2", "MAPE", 12.34)

    captured = capsys.readouterr()
    expected_output = "[LOG] Model: Llama-2 | Metric: MAPE | Score: 12.3400\n"
    
    with open(file_logger.file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    assert captured.out == expected_output, "ConsoleLogger did not receive notification correctly."
    assert len(lines) == 1, "FileLogger should log only one entry."
    assert lines[0].strip() == "Model: Llama-2 | Metric: MAPE | Score: 12.3400"
