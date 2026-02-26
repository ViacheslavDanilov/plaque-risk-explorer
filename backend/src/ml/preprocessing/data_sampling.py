import logging
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SMOTENC

from data_loader import MyDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataSampling:
    def __init__(self, X, Y, sampling_mode: str = "append"):
        """
        Initialize DataSampling object.
        Args:
            X: Feature matrix (pandas DataFrame).
            Y: Target vector (pandas Series).
            sampling_mode: Sampling mode ('append' or 'new'). Default is 'append'.
        """
        self.X = X
        self.Y = Y
        self.x = None
        self.y = None
        self.sampler = None
        self.sampler_method = None
        self.sampling_mode = sampling_mode  # 'append' (default) or 'new'

    def get_sampler(self, method: str = "under", sampling_strategy: str = "not minority", categorical_features=None):
        """
        Initialize the sampler object based on the specified method and strategy.
        Args:
            method: Sampling method ('under', 'over', 'smote', 'adasyn').
            sampling_strategy: Strategy for sampling (e.g., 'auto', 'not minority').
            categorical_features: List of categorical feature indices for SMOTENC.
        """
        self.sampler_method = method
        match method:
            case "under":
                self.sampler = RandomUnderSampler(sampling_strategy=sampling_strategy)
                logger.info(f"RandomUnderSampler initialized with sampling_strategy='{sampling_strategy}'")
            case "over":
                self.sampler = RandomOverSampler(sampling_strategy=sampling_strategy)
                logger.info(f"RandomOverSampler initialized with sampling_strategy='{sampling_strategy}'")
            case "smote":
                if categorical_features is not None and len(categorical_features) > 0:
                    self.sampler = SMOTENC(
                        categorical_features=categorical_features,
                        sampling_strategy=sampling_strategy,
                        random_state=42,
                        k_neighbors=2
                    )
                    logger.info(f"SMOTENC initialized with categorical_features={categorical_features} and sampling_strategy='{sampling_strategy}'")
                else:
                    self.sampler = SMOTE(sampling_strategy=sampling_strategy)
                    logger.info(f"SMOTE initialized with sampling_strategy='{sampling_strategy}'")
            case "adasyn":
                self.sampler = ADASYN(sampling_strategy=sampling_strategy)
                logger.info(f"ADASYN initialized with sampling_strategy='{sampling_strategy}'")
            case _:
                raise ValueError(f"Unsupported sampling method: {method}")

    def resample_data(self):
        """
        Apply the sampler to the feature matrix and target vector.
        Stores the resampled features and targets in self.x and self.y.
        """
        if self.sampler is not None:
            # log original distribution
            if self.Y is not None:
                orig_counts = self.Y.value_counts()
                logger.info(f"Original target distribution: {orig_counts.to_dict()}")
            x_res, y_res = self.sampler.fit_resample(self.X, self.Y)
            self.x = x_res
            self.y = y_res
            res_counts = y_res.value_counts()
            logger.info(
                f"Data resampled using {self.sampler_method}-sampling. "
                f"Resampled data shape: {x_res.shape}, {y_res.shape}. "
                f"New target distribution: {res_counts.to_dict()}"
            )
        else:
            raise ValueError("Sampler not set. Please call get_sampler() to set the sampling method before plotting.")

    def export_csv(self, path: str, include_index: bool = False, val_round: int = None):
        """
        Export the synthetic dataset to a CSV file.
        Combines features and target into a single DataFrame and writes it out.
        Args:
            path: Output file path.
            include_index: Whether to include row indices in the CSV.
            val_round: Number of decimal places to round numeric values.
        """
        if self.x is None or self.y is None:
            raise ValueError("No resampled data available. Call resample_data() first.")

        # Формируем итоговый датасет в зависимости от sampling_mode
        if self.sampling_mode == "append":
            # Просто сохраняем все resampled данные (исходные + синтетические)
            df = pd.concat([pd.DataFrame(self.x), pd.Series(self.y, name="target")], axis=1)
        elif self.sampling_mode == "new":
            # Оставляем только синтетические (исключаем исходные)
            # Для oversampling: новые индексы = индексы, которых не было в исходных
            if hasattr(self.sampler, 'sample_indices_'):
                orig_len = len(self.X)
                synthetic_mask = [i for i in self.sampler.sample_indices_ if i >= orig_len]
                if synthetic_mask:
                    synthetic_x = self.x.iloc[synthetic_mask] if hasattr(self.x, 'iloc') else self.x[synthetic_mask]
                    synthetic_y = self.y.iloc[synthetic_mask]
                    df = pd.concat([pd.DataFrame(synthetic_x), pd.Series(synthetic_y, name="target")], axis=1)
                else:
                    logger.warning("No synthetic samples found for 'new' mode. Exporting empty file.")
                    df = pd.DataFrame()
            else:
                logger.warning("Sampler does not provide sample_indices_. Exporting all resampled data.")
                df = pd.concat([pd.DataFrame(self.x), pd.Series(self.y, name="target")], axis=1)
        else:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")

        # Округляем все числовые значения до указанного количества знаков после запятой
        if val_round is not None:
            df = df.apply(lambda col: col.round(val_round) if col.dtype.kind in 'fc' else col)
        df.to_csv(path, index=include_index)
        logger.info(f"Data exported to '{path}', mode={self.sampling_mode}, include_index={include_index}")

    def data_plot(self, ncols: int = 2, figsize: tuple = (12, 6), autopct: str = "%1.1f%%"):
        """
        Plot pie charts of target distribution before and after sampling.
        Args:
            ncols: Number of columns in the subplot.
            figsize: Figure size.
            autopct: Format for pie chart labels.
        Returns:
            fig, axs: Matplotlib figure and axes objects.
        """
        if self.Y is not None and self.y is not None:
            fig, axs = plt.subplots(ncols=ncols, figsize=figsize)
            y_source = self.Y
            y_res = self.y
            
            y_source.value_counts().plot.pie(autopct=autopct, ax=axs[0])
            axs[0].set_title(f"Original data")
            y_res.value_counts().plot.pie(autopct=autopct, ax=axs[1])
            axs[1].set_title(f"{self.sampler_method}-sampling")
            return fig, axs
        else:
            raise ValueError("Sampler not set. Please call get_sampler() to set the sampling method before plotting.")
        
    def synthetic_comparison_plot(self, features=None, figsize=(12, 8)):
        """
        Plot overlay histograms comparing feature distributions between real and synthetic data.
        In 'new' mode: compares original and synthetic data.
        In 'append' mode: compares original and resampled data.
        Args:
            features: List of feature names to compare (default: age, cholesterol_level, syntax_score, angina_functional_class, lumen_area).
            figsize: Figure size.
        Returns:
            fig, axs: Matplotlib figure and axes objects.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        if features is None:
            features = ["age", "cholesterol_level", "syntax_score", "angina_functional_class", "lumen_area"]
        n_features = len(features)
        fig, axs = plt.subplots(n_features, 1, figsize=figsize)
        if n_features == 1:
            axs = [axs]
        for i, feat in enumerate(features):
            # Получаем значения для исходных и синтетических
            real_vals = self.X[feat] if hasattr(self.X, '__getitem__') else None
            syn_vals = None
            if self.sampling_mode == "new":
                # Только синтетические
                if hasattr(self.sampler, 'sample_indices_'):
                    orig_len = len(self.X)
                    synthetic_mask = [idx for idx in self.sampler.sample_indices_ if idx >= orig_len]
                    syn_vals = self.x.iloc[synthetic_mask][feat] if hasattr(self.x, 'iloc') else self.x[synthetic_mask, self.x.columns.get_loc(feat)]
                else:
                    syn_vals = self.x[feat] if hasattr(self.x, '__getitem__') else None
                axs[i].hist(real_vals, bins=20, alpha=0.5, label="Real", color="blue")
                axs[i].hist(syn_vals, bins=20, alpha=0.5, label="Synthetic", color="orange")
                axs[i].set_title(f"{feat}: Real vs Synthetic (new mode)")
            else:
                syn_vals = self.x[feat] if hasattr(self.x, '__getitem__') else None
                axs[i].hist(real_vals, bins=20, alpha=0.5, label="Real", color="blue")
                axs[i].hist(syn_vals, bins=20, alpha=0.5, label="Resampled", color="green")
                axs[i].set_title(f"{feat}: Real vs Resampled (append mode)")
            axs[i].legend()
            axs[i].set_xlabel(feat)
            axs[i].set_ylabel("Count")
        fig.tight_layout()
        return fig, axs


if __name__ == "__main__":
    # Example usage
    sampling_method = "smote"  # "over" or "under" or "smote" or "adasyn"
    sampling_strategy = "auto"  # "not majority", "all", "auto", or a dict specifying the desired number of samples for each class
    dataloader = MyDataLoader(data_path="backend/data/features.csv")
    dataloader.load_data()
    dataloader.impute_missing_values(n_neighbors=3)

    X, Y = dataloader.get_data_features_and_target(target_column="adverse_outcome")
    # Определяем категориальные признаки для SMOTENC
    import pandas as pd
    if isinstance(X, pd.DataFrame):
        categorical_features = [i for i, dt in enumerate(X.dtypes) if dt == 'object' or str(dt).startswith('category')]
    else:
        categorical_features = []
    my_data_sampler = DataSampling(
        X = X,
        Y = Y,
        sampling_mode = "new"  # default
    )
    my_data_sampler.get_sampler(method=sampling_method, sampling_strategy=sampling_strategy, categorical_features=categorical_features)
    my_data_sampler.resample_data()
    my_data_sampler.export_csv(path=f"backend/data/resampled_data_{my_data_sampler.sampler_method}.csv", include_index=False)
    fig, axs = my_data_sampler.data_plot()
    fig.savefig(f"backend/data/sampling_plot_{my_data_sampler.sampler_method}.png")
    fig, axs = my_data_sampler.synthetic_comparison_plot()
    fig.savefig(f"backend/data/synthetic_comparison_{my_data_sampler.sampler_method}.png")
    logger.info(f"Sampling process completed using method '{sampling_method}' with strategy '{sampling_strategy}'.")