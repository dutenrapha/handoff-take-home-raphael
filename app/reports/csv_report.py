import pandas as pd
from reports.base_report import BaseReport

class CSVReport(BaseReport):
    """Generates a CSV report from evaluation results."""

    def generate_report(self, results: pd.DataFrame, output_path: str):
        """
        Saves the evaluation results to a CSV file.

        Args:
            results (pd.DataFrame): The evaluation results.
            output_path (str): Path to save the report.
        """
        results.to_csv(output_path, index=False, encoding="utf-8")
        print(f"[REPORT] CSV report saved at: {output_path}")
