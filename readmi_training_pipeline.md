# Training Pipeline: Methodology and Workflow

## Overview

This document describes the training pipeline for adverse outcome prediction using clinical and morphological data. The core of the pipeline is the `bootstrap_validation` method, which provides robust model evaluation and optimism correction.

## Data Sources

- **Training data:** Synthetic dataset generated via advanced sampling (see `readmi_synthetic_data.md`).
- **Testing data:** Real, untouched patient data for unbiased evaluation.

## Main Steps

1. **Data Preparation:**
   - Synthetic data is used for model training to address class imbalance and increase sample size.
   - Real data is reserved for final testing and validation.
2. **Model Training:**
    - The core step is the `bootstrap_validation` method, which implements bootstrap validation. This statistical approach repeatedly resamples the training set with replacement (bootstrap samples). For each iteration:
       1. A new training subsample (in-bag) is created, and the remaining samples (not included in the in-bag set) form the out-of-bag (OOB) set for testing.
       2. The model is trained on the in-bag data and evaluated on the OOB samples, providing an independent performance estimate without a separate test split.
       3. For each iteration, performance metrics (ROC-AUC, PR-AUC) are computed for both the training and OOB sets.
       4. Final performance is optimism-corrected: the average difference between in-bag and OOB metrics is subtracted from the model's performance on the full training set.
    - The model used is AutoGluon TabPFNv2—a state-of-the-art AutoML framework for tabular data. TabPFNv2 leverages advanced neural architectures and ensembles for classification and regression tasks.
    - All steps, parameters, model paths, and intermediate/final results are logged using Python's built-in logging. This ensures transparency, reproducibility, and ease of analysis.
3. **Performance Estimation:**
   - For each bootstrap iteration, ROC-AUC and PR-AUC are computed for both apparent (in-bag) and test (OOB) sets.
   - Optimism correction is applied: the difference between apparent and test performance is averaged and subtracted from the final model's performance on the full training set.
   - The corrected metrics provide an unbiased estimate of model generalization.
4. **Leaderboard and Logging:**
   - After training, a leaderboard of model performance is saved as an Excel file in the model directory.
   - All key steps, paths, and results are logged for reproducibility.

## Key Advantages

- **Robustness:** Bootstrap validation provides a more reliable estimate of model performance, especially with small or imbalanced datasets.
- **Transparency:** All steps are logged, and results are saved for further analysis.
- **Separation of train/test:** Synthetic data is used for training, while real data is used for final evaluation, preventing data leakage.

## Example Workflow

1. Generate synthetic data (see `readmi_synthetic_data.md`).
2. Train model using `bootstrap_validation` on synthetic data.
3. Evaluate on real data and save leaderboard.

---
*For details, see the code and logs. Replace placeholders with actual results and figures as needed.*

## Log-based Results and Statistics

### Log Summary

- Data loaded from: backend/data/resampled_data_smote.csv (synthetic, train)
- Data loaded from: backend/data/features.csv (real, test)
- Bootstrap validation started: 300 iterations
- Progress logged every 25 iterations
- Example leaderboard (AutoGluon):

| model                | score_test | score_val | eval_metric | pred_time_test | pred_time_val | fit_time | ... |
|----------------------|------------|-----------|-------------|---------------|--------------|----------|-----|
| RealTabPFN-v2.5      | 1.0        | 1.0       | accuracy    | 0.246         | 0.220        | 0.088    | ... |
| WeightedEnsemble_L2  | 1.0        | 1.0       | accuracy    | 0.247         | 0.220        | 0.089    | ... |

### Final Results

- Apparent ROC-AUC: 1.000
- Optimism: 0.001
- Corrected ROC-AUC: 0.999
- Apparent PR-AUC: 1.000
- Optimism: 0.001
- Corrected PR-AUC: 0.999

All results, leaderboards, and logs are saved for reproducibility and further analysis.
