import gc
import torch
import logging
import numpy as np
import datetime

from sklearn.model_selection import train_test_split, LeaveOneOut, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import resample
from autogluon.tabular import TabularDataset, TabularPredictor

from backend.src.ml.preprocessing.data_loader import MyDataLoader

log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_file = "backend/logs/training.log"

file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.setFormatter(log_formatter)
logger.addHandler(file_handler)


class AdverseOutcomeModel:
    """Placeholder: class for loading and using the adverse outcome model."""
    def __init__(self, model_name='REALTABPFN-V2.5', data_synthetic_method='adasyn'):
        self.tabpfnv2_predictor = None
        self.loo = LeaveOneOut()
        self.train_data = None
        self.test_data = None
        self.model_name = model_name
        self.data_synthetic_method = data_synthetic_method
        self.model_path = f'backend/models/tabpfnv2_model/{self.data_synthetic_method}/{self.model_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'

    def predict(self, data):
        tabpfnv2_predictions = self.tabpfnv2_predictor.predict_proba(data)
        logger.info(f"TabPFNv2 predictions head:\n{tabpfnv2_predictions.head()}")
        self.tabpfnv2_predictor.leaderboard(data, silent=True)
        logger.info("Model evaluation complete.")

    def convert_datset_split(self, data):
        # Convert to TabularDataset
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        self.train_data = TabularDataset(train_data)
        self.test_data = TabularDataset(test_data)
        logger.info(f"Converted dataset with shapes: train={self.train_data.shape}, test={self.test_data.shape}")
    
    def convert_datset(self, train_data, test_data):
        # Convert to TabularDataset
        self.train_data = TabularDataset(train_data)
        self.test_data = TabularDataset(test_data)
        logger.info(f"Converted dataset with shapes: train={self.train_data.shape}, test={self.test_data.shape}")
    
    def train_stratified(self, target_column: str, n_splits: int = 5):
        """
        Train using Stratified K-Fold Cross-Validation.
        Collects predictions for all samples in each fold for reliable metrics.
        """
        logger.info(f"Starting {n_splits}-fold Stratified CV...")

        X = self.train_data.drop(columns=[target_column])
        y = self.train_data[target_column]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        preds, true = [], []

        for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            fold_num = i + 1
            logger.info(f"--- Fold {fold_num}/{n_splits} ---")

            train_df = self.train_data.iloc[train_idx]
            test_df = self.train_data.iloc[test_idx]
            
            # Unique path for each fold to avoid overwriting
            fold_model_path = f"{self.model_path}_fold_{fold_num}"

            predictor = TabularPredictor(
                label=target_column,
                eval_metric="roc_auc",  # roc_auc log_loss
                verbosity=0,
                path=fold_model_path
            )

            predictor.fit(
                train_data=TabularDataset(train_df),
                hyperparameters={self.model_name: {}},
                presets="best_quality"
            )
            
            # Predict for ALL samples in the test fold
            proba = predictor.predict_proba(TabularDataset(test_df)).iloc[:, 1].values
            preds.extend(proba)
            true.extend(test_df[target_column].values)
            
            logger.info(f"Fold {fold_num} completed.")
            self.evaluate_metrics(true, preds)
            self.tabpfnv2_predictor = predictor # Keep last predictor

        logger.info("StratifiedKFold finished. Final Cross-Validation Evaluation:")
        self.evaluate_metrics(true, preds)
        
        leaderboard = self.tabpfnv2_predictor.leaderboard(self.test_data, silent=False)
        leaderboard.to_excel(f"{self.model_path}_fold_{n_splits}/leaderboard.xlsx", index=False)
        logger.info(f"Leaderboard saved to: {self.model_path}_fold_{n_splits}/leaderboard.xlsx")
        logger.info(f"Leaderboard:\n{leaderboard}")


    def evaluate_metrics(self, true, preds):
        auc = roc_auc_score(true, preds)
        pr = average_precision_score(true, preds)

        logger.info(f"ROC-AUC: {auc}")
        logger.info(f"PR-AUC: {pr}")


if __name__ == "__main__":
    data_synthetic_method = "smote"  # Options: "adasyn", "smote"
    # 1. Load Synthetic Train Data
    my_data_loader = MyDataLoader(data_path=f"backend/data/resampled_data_{data_synthetic_method}.csv")
    my_data_loader.load_data()
    my_data_loader.impute_missing_values(n_neighbors=3)
    train_data = my_data_loader.data

    # 2. Load Real Test Data
    my_data_loader = MyDataLoader(data_path="backend/data/features.csv")
    my_data_loader.load_data()
    my_data_loader.impute_missing_values(n_neighbors=3)
    test_data = my_data_loader.data

    # 3. Initialize Model
    adverse_outcome_model = AdverseOutcomeModel(data_synthetic_method=data_synthetic_method)
    adverse_outcome_model.convert_datset(train_data, test_data)
    
    # 4. Run Cross-Validation on Synthetic Data (Internal Validation)
    logger.info("Running Cross-Validation on Synthetic Data...")
    adverse_outcome_model.train_stratified(target_column="adverse_outcome", n_splits=5)
    
    # 5. Final Evaluation on Real Data
    logger.info("Performing Final Evaluation on REAL data (features.csv)...")
    
    # Check class distribution
    counts = test_data["adverse_outcome"].value_counts()
    logger.info(f"Test Set Class Distribution:\n{counts}")
    
    if len(counts) < 2:
        logger.warning("CRITICAL: Test set has only ONE class. Metrics are meaningless.")
    
    final_probas = adverse_outcome_model.tabpfnv2_predictor.predict_proba(test_data).iloc[:, 1]
    final_auc = roc_auc_score(test_data["adverse_outcome"], final_probas)
    final_pr = average_precision_score(test_data["adverse_outcome"], final_probas)
    
    logger.info(f"====== REAL WORLD TEST RESULTS ======")
    logger.info(f"ROC-AUC on Real Data: {final_auc:.4f}")
    logger.info(f"PR-AUC on Real Data: {final_pr:.4f}")
    logger.info("=====================================")

    # 6. Investigate Feature Importance
    try:
        logger.info("Calculating Feature Importance (this may take a while)...")
        importance = adverse_outcome_model.tabpfnv2_predictor.feature_importance(test_data)
        logger.info(f"Feature Importance:\n{importance}")
        importance.to_excel(f"{adverse_outcome_model.model_path}_feature_importance.xlsx", index=False)
    except Exception as e:
        logger.warning(f"Could not calculate feature importance: {e}")

    # 7. Check for potential overlap/leakage
    # Check if any rows in test_data are identical to train_data (ignoring target)
    train_no_target = train_data.drop(columns=["adverse_outcome"])
    test_no_target = test_data.drop(columns=["adverse_outcome"])
    
    overlap = test_no_target.merge(train_no_target, how='inner')
    if not overlap.empty:
        logger.warning(f"CRITICAL LEAKAGE: Found {len(overlap)} rows in Test Data that also exist in Train Data!")
    else:
        logger.info("No exact row duplicates found between Train and Test sets.")
