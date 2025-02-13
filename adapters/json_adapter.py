import os
import json
import pandas as pd
from adapters.base_adapter import BaseAdapter

class JSONAdapter(BaseAdapter):
    """
    Adapter to transform JSON data into a standardized Pandas DataFrame for ground truth,
    or extract model outputs when the JSON represents model output data.
    """

    def __init__(self, file_path: str):
        """
        Initializes the JSONAdapter with the file path.

        Args:
            file_path (str): Path to the JSON file.
        """
        self.file_path = file_path
        self.data = self._load_json()

    def _load_json(self) -> dict:
        """
        Loads JSON data from the given file path.

        Returns:
            dict: Parsed JSON data.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        with open(self.file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts JSON data (assumed to be a ground truth file) into a Pandas DataFrame.

        Expected ground truth JSON structure:
            test_n: int,
            input: str,
            rows: list of objects,
            totalCostUsd: number,
            ... (each row contains keys such as 'sectionName', 'qty', 'rateUsd', 
                'rowTotalCostUsd', 'label', 'uom', 'category')
        
        Returns:
            pd.DataFrame: A DataFrame built from the 'rows' key.
        """
        if "rows" not in self.data:
            raise ValueError(f"Invalid JSON format: 'rows' key missing in {self.file_path}")
        return pd.DataFrame(self.data["rows"])

    def to_model_outputs(self) -> list:
        """
        Converts a model output JSON file into a list of outputs. The expected model output JSON
        structure is:
            {
                "estimate_preds": [
                    {
                        "valid_file_name": str,        # Name of the ground truth file used
                        "rows": list of objects,         # Generated line-item estimates (same format as ground truth)
                        "time_to_estimate_sec": number   # Time taken to generate the estimate
                    },
                    ... (more predictions, even with duplicate valid_file_name values)
                ]
            }

        Returns:
            list: A list of dictionaries representing each model output.
                  Each dictionary contains:
                    - 'valid_file_name': str,
                    - 'df': pd.DataFrame (converted from 'rows'),
                    - 'time_to_estimate_sec': number
        """
        if "estimate_preds" not in self.data:
            raise ValueError(f"Invalid model output file: 'estimate_preds' key not found in {self.file_path}")

        outputs = []
        for pred in self.data["estimate_preds"]:
            # Validate that all required keys are present.
            required_keys = ["valid_file_name", "rows", "time_to_estimate_sec"]
            missing_keys = [key for key in required_keys if key not in pred]
            if missing_keys:
                print(f"Warning: Skipping prediction in '{self.file_path}' due to missing keys: {missing_keys}")
                continue
            df = pd.DataFrame(pred["rows"])
            outputs.append({
                "valid_file_name": pred["valid_file_name"],
                "df": df,
                "time_to_estimate_sec": pred["time_to_estimate_sec"]
            })
        return outputs
