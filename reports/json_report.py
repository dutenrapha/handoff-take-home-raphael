import pandas as pd
import json
from reports.base_report import BaseReport

class JSONReport(BaseReport):
    """Generates a JSON report from evaluation results."""

    def generate_report(self, results: pd.DataFrame, output_path: str):
        """
        Saves the evaluation results to a JSON file.

        Args:
            results (pd.DataFrame): The evaluation results.
            output_path (str): Path to save the report.
        """
        results_dict = results.to_dict(orient="records")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(results_dict, file, indent=4)

        print(f"[REPORT] JSON report saved at: {output_path}")
