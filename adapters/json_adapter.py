import os
import json
import pandas as pd
from adapters.base_adapter import BaseAdapter

class JSONAdapter(BaseAdapter):
    """Adapter to transform JSON data into a standardized Pandas DataFrame."""

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
        Converts the JSON data into a Pandas DataFrame.

        Returns:
            pd.DataFrame: A standardized DataFrame containing line items.
        """
        if "rows" not in self.data:
            raise ValueError(f"Invalid JSON format: 'rows' key missing in {self.file_path}")

        return pd.DataFrame(self.data["rows"])
