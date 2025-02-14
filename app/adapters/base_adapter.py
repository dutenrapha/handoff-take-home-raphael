from abc import ABC, abstractmethod
import pandas as pd

class BaseAdapter(ABC):
    """Abstract base class for adapters that transform data into a standardized format."""

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the input data into a Pandas DataFrame.

        Returns:
            pd.DataFrame: The transformed dataset.
        """
        pass
