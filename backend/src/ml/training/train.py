import gc
import torch
import logging
import numpy as np

from sklearn.model_selection import train_test_split, LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import resample
from autogluon.tabular import TabularDataset, TabularPredictor

from backend.src.ml.preprocessing.data_loader import MyDataLoader

log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_file = "backend/logs/training.log"

file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


class AdverseOutcomeModel:
    """Placeholder: class for loading and using the adverse outcome model."""
    def __init__(self):
        self.tabpfnv2_predictor = None
        self.loo = LeaveOneOut()
        self.train_data = None
        self.test_data = None

    def predict(self, data):
        tabpfnv2_predictions = self.tabpfnv2_predictor.predict_proba(data)
        logger.info(f"TabPFNv2 predictions head:\n{tabpfnv2_predictions.head()}")
        self.tabpfnv2_predictor.leaderboard(data, silent=True)
        logger.info("Model evaluation complete.")

    def convert_datset(self, data):
        # Convert to TabularDataset
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        self.train_data = TabularDataset(train_data)
        self.test_data = TabularDataset(test_data)
        logger.info(f"Converted dataset with shapes: train={self.train_data.shape}, test={self.test_data.shape}")

    def train_model(self, target_column: str):
        """
        Train TabPFNv2 model and output leaderboard after training.
        """
        logger.info("Training TabPFNv2 on Input dataset...")
        self.tabpfnv2_predictor = TabularPredictor(
            label=target_column,
            eval_metric="roc_auc",
            path='backend/models/tabpfnv2_model'
        )
        self.tabpfnv2_predictor.fit(
            self.train_data,
            hyperparameters={
                'REALTABPFN-V2.5': {},
            },
            presets="best_quality",
        )
        logger.info("Model training complete. Evaluating on test set...")
        leaderboard = self.tabpfnv2_predictor.leaderboard(self.test_data, silent=False)
        logger.info(f"Leaderboard:\n{leaderboard}")


    def bootstrap_validation(self, data, target_column: str, n_bootstrap: int = 300):
        logger.info("Starting Bootstrap validation...")

        X = data.drop(columns=[target_column])
        y = data[target_column]

        apparent_aucs = []
        test_aucs = []
        apparent_pr = []
        test_pr = []

        for i in range(n_bootstrap):
            boot_idx = resample(range(len(data)), replace=True, n_samples=len(data))
            oob_idx = list(set(range(len(data))) - set(boot_idx))

            if len(oob_idx) < 3:
                continue

            train_df = data.iloc[boot_idx]
            if train_df[target_column].nunique() < 2:
                continue

            test_df = data.iloc[oob_idx]
            if test_df[target_column].nunique() < 2:
                continue

            self.tabpfnv2_predictor = TabularPredictor(
                label=target_column,
                eval_metric="log_loss",
                verbosity=0
            )

            self.tabpfnv2_predictor.fit(
                train_data=train_df,
                hyperparameters={'REALTABPFN-V2.5': {}},
                num_bag_folds=0,
                num_stack_levels=0,
            )

            p_train = self.tabpfnv2_predictor.predict_proba(train_df).iloc[:, 1]
            auc_app = roc_auc_score(train_df[target_column], p_train)
            pr_app = average_precision_score(train_df[target_column], p_train)

            # test performance
            p_test = self.tabpfnv2_predictor.predict_proba(test_df).iloc[:, 1]
            auc_test = roc_auc_score(test_df[target_column], p_test)
            pr_test = average_precision_score(test_df[target_column], p_test)

            apparent_aucs.append(auc_app)
            test_aucs.append(auc_test)
            apparent_pr.append(pr_app)
            test_pr.append(pr_test)

            if i % 25 == 0:
                logger.info(f"Bootstrap {i}/{n_bootstrap}")

        # optimism correction
        optimism_auc = np.mean(np.array(apparent_aucs) - np.array(test_aucs))
        optimism_pr = np.mean(np.array(apparent_pr) - np.array(test_pr))

        # apparent performance на всей выборке
        self.tabpfnv2_predictor = TabularPredictor(label=target_column, verbosity=0)
        self.tabpfnv2_predictor.fit(data, hyperparameters={'REALTABPFN-V2.5': {}})

        p_full = self.tabpfnv2_predictor.predict_proba(data).iloc[:, 1]
        auc_full = roc_auc_score(y, p_full)
        pr_full = average_precision_score(y, p_full)

        leaderboard = self.tabpfnv2_predictor.leaderboard(self.test_data, silent=False)
        logger.info(f"Leaderboard:\n{leaderboard}")

        corrected_auc = auc_full - optimism_auc
        corrected_pr = pr_full - optimism_pr

        logger.info("====== FINAL RESULTS ======")
        logger.info(f"Apparent ROC-AUC: {auc_full:.3f}")
        logger.info(f"Optimism: {optimism_auc:.3f}")
        logger.info(f"Corrected ROC-AUC: {corrected_auc:.3f}")

        logger.info(f"Apparent PR-AUC: {pr_full:.3f}")
        logger.info(f"Optimism: {optimism_pr:.3f}")
        logger.info(f"Corrected PR-AUC: {corrected_pr:.3f}")

    
    def train_stratified(self, data, target_column: str):

        preds, true = [], []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Fold {i+1}/5")

            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            self.tabpfnv2_predictor = TabularPredictor(
                label=target_column,
                eval_metric="average_precision",
                verbosity=0
            )

            self.tabpfnv2_predictor.fit(
                train_data=train_data,
                num_bag_folds=None,
                num_stack_levels=0,
                hyperparameters="very_light",
                calibrate=False 
            )
            
            proba = self.tabpfnv2_predictor.predict_proba(test_data).iloc[:, 1].values[0]
            preds.append(proba)
            true.append(test_data[target_column].values[0])
            logger.info(f"Fold {i+1} - True: {true[-1]}, Predicted Probability: {proba:.4f}")
            self.evaluate_loocv(true, preds)

            leaderboard = self.tabpfnv2_predictor.leaderboard(self.test_data, silent=False)
            logger.info(f"Leaderboard:\n{leaderboard}")
        
        logger.info("StratifiedKFold finished")
        self.evaluate_loocv(true, preds)

    
    def train_loocv(self, data, target_column: str):
        logger.info("Starting Leave-One-Out CV...")

        X = data.drop(columns=[target_column])
        preds, true = [], []

        for i, (train_idx, test_idx) in enumerate(self.loo.split(X)):
            logger.info(f"Fold {i+1}/{len(X)}")

            train_df = data.iloc[train_idx]
            test_df = data.iloc[test_idx]

            self.tabpfnv2_predictor = TabularPredictor(
                label=target_column,
                eval_metric="log_loss", # roc_auc
                verbosity=0
            )

            self.tabpfnv2_predictor.fit(
                TabularDataset(train_df),
                hyperparameters={'REALTABPFN-V2.5': {}},
                ag_args_fit={"num_gpus": 0},
                presets="best_quality"
            )

            proba = self.tabpfnv2_predictor.predict_proba(TabularDataset(test_df)).iloc[0, 1]

            preds.append(proba)
            true.append(test_df[target_column].values[0])

            logger.info(f"Fold {i+1} - True: {true[-1]}, Predicted Probability: {proba:.4f}")
            self.evaluate_loocv(true, preds)

            leaderboard = self.tabpfnv2_predictor.leaderboard(self.test_data, silent=False)
            logger.info(f"Leaderboard:\n{leaderboard}")

        logger.info("LOOCV finished")
        self.evaluate_loocv(true, preds)


    def evaluate_loocv(self, true, preds):
        auc = roc_auc_score(true, preds)
        pr = average_precision_score(true, preds)

        logger.info(f"ROC-AUC: {auc}")
        logger.info(f"PR-AUC: {pr}")


if __name__ == "__main__":
    my_data_loader = MyDataLoader(data_path="backend/data/resampled_data_smote.csv")
    my_data_loader.load_data()
    # my_data_loader.impute_missing_values(n_neighbors=3)
    train_data = my_data_loader.data

    my_data_loader = MyDataLoader(data_path="backend/data/features.csv")
    my_data_loader.load_data()
    # my_data_loader.impute_missing_values(n_neighbors=3)
    test_data = my_data_loader.data
    test_data["target"] = test_data["adverse_outcome"]

    # print(data["adverse_outcome"].value_counts())
    # print(data.groupby("adverse_outcome").mean(numeric_only=True))

    # adverse_outcome_model = AdverseOutcomeModel()
    # # adverse_outcome_model.convert_datset(data)
    # # adverse_outcome_model.train_model(target_column="adverse_outcome")
    # adverse_outcome_model.train_loocv(data, target_column="adverse_outcome")
    adverse_outcome_model = AdverseOutcomeModel()
    adverse_outcome_model.train_data = train_data
    adverse_outcome_model.test_data = test_data
    adverse_outcome_model.train_stratified(train_data, target_column="target")
    # adverse_outcome_model.bootstrap_validation(data, target_column="adverse_outcome", n_bootstrap=300)
    # adverse_outcome_model.bootstrap_validation(train_data, target_column="target", n_bootstrap=300)

    # # adverse_outcome_model.predict(adverse_outcome_model.test_data)
