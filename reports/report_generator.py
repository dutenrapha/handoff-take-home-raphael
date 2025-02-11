from reports.csv_report import CSVReport
from reports.json_report import JSONReport
import pandas as pd

class ReportGenerator:
    """Handles the creation of evaluation reports in different formats."""

    def __init__(self, format: str = "csv"):
        """
        Initializes the report generator.

        Args:
            format (str): The format of the report ("csv" or "json").
        """
        self.format = format.lower()
        self.report = self._get_report_instance()

    def _get_report_instance(self):
        """Returns the appropriate report generator based on the format."""
        if self.format == "csv":
            return CSVReport()
        elif self.format == "json":
            return JSONReport()
        else:
            raise ValueError(f"Unsupported report format: {self.format}")

    def generate(self, results: pd.DataFrame, output_path: str):
        """
        Generates a report based on the evaluation results.

        Args:
            results (pd.DataFrame): The evaluation results.
            output_path (str): Path to save the report.
        """
        self.report.generate_report(results, output_path)
