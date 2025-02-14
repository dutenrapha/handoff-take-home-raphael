from abc import ABC, abstractmethod
import pandas as pd

class BaseReport(ABC):
    """Abstract base class for generating evaluation reports."""

    @abstractmethod
    def generate_report(self, results: pd.DataFrame, output_path: str):
        """
        Generates a report based on model evaluation results.

        Args:
            results (pd.DataFrame): The evaluation results.
            output_path (str): Path to save the report.
        """
        pass
