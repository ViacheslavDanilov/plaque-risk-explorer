import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MyDataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        # Implement your data loading logic here
        self.data = pd.read_csv(self.data_path)
        logger.info(f"Data loaded from path: {self.data_path}")
        return self.data
    
    def get_data_features_and_target(self, target_column: str):
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() before getting features and target.")
        
        X = self.data.drop(columns=[target_column])
        Y = self.data[target_column]
        logger.info(f"Features and target extracted from data. Features shape: {X.shape}, Target shape: {Y.shape}")
        return X, Y

    def describe_target(self, target_column: str) -> pd.Series:
        """Return a series with counts for each class in the target column.

        Also logs the number of unique classes and their distribution.
        Raises ``ValueError`` if data has not been loaded.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() before describing the target.")

        if target_column not in self.data.columns:
            raise KeyError(f"Column '{target_column}' not found in data.")

        counts = self.data[target_column].value_counts(dropna=False)
        num_classes = counts.shape[0]
        logger.info(
            "Target '%s' has %d classes. Distribution:\n%s",
            target_column,
            num_classes,
            counts.to_string(),
        )
        return counts
    

if __name__ == "__main__":
    # Example usage
    my_data_loader = MyDataLoader(data_path="backend/data/features.csv")
    data = my_data_loader.load_data()
    X, Y = my_data_loader.get_data_features_and_target(target_column="adverse_outcome")
    my_data_loader.describe_target(target_column="adverse_outcome")