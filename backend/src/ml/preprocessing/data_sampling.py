import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_loader import MyDataLoader
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataSampling:
    def __init__(self, X, Y, data_types, sampling_mode: str = "append"):
        """
        Initialize DataSampling object.
        Args:
            X: Feature matrix (pandas DataFrame).
            Y: Target vector (pandas Series).
            data_types: Original dtypes of the data.
            sampling_mode: Sampling mode ('append' or 'new'). Default is 'append'.
        """
        self.X = X
        self.Y = Y
        self.orig_len = len(X)
        self.x = None
        self.y = None
        self.data_types = data_types
        self.sampler = None
        self.sampler_method = None
        self.sampling_mode = sampling_mode  # 'append' (default) or 'new'

    def check_intersection_with_original(self):
        """
        Проверяет, есть ли пересечение между синтетическими и исходными строками (по всем признакам и target).
        Возвращает количество и список пересечений (до 10 строк).
        """
        if self.x is None or self.y is None:
            raise ValueError("No resampled data available. Call resample_data() first.")
        orig_df = pd.concat(
            [self.X.reset_index(drop=True), self.Y.reset_index(drop=True)],
            axis=1,
        )
        orig_df = orig_df.astype(self.data_types.to_dict())
        synth_df = pd.concat(
            [self.x.reset_index(drop=True), self.y.reset_index(drop=True)],
            axis=1,
        )
        synth_df = synth_df.astype(self.data_types.to_dict())
        # Преобразуем в кортежи для сравнения
        orig_set = set(tuple(row) for row in orig_df.values)
        synth_set = set(tuple(row) for row in synth_df.values)
        intersection = orig_set & synth_set
        return len(intersection), list(intersection)[:10]

    def _fill_categorical_random(self, synth_df, orig_df, categorical_cols):
        """
        Заполняет категориальные признаки в synth_df случайными значениями из orig_df.
        """
        for col in categorical_cols:
            synth_df[col] = np.random.choice(
                orig_df[col].dropna().values,
                size=len(synth_df),
            )
        return synth_df

    def get_sampler(
        self,
        method: str = "under",
        sampling_strategy: str = "not minority",
        categorical_features=None,
        k_neighbors: int = 5,
    ):
        """
        Initialize the sampler object based on the specified method and strategy.
        Args:
            method: Sampling method ('under', 'over', 'smote', 'adasyn').
            sampling_strategy: Strategy for sampling (e.g., 'auto', 'not minority', or dict).
            categorical_features: List of categorical feature indices for SMOTENC.
            k_neighbors: Number of nearest neighbors to use for over-sampling (e.g. for SMOTE).
        """
        self.sampler_method = method
        match method:
            case "under":
                self.sampler = RandomUnderSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=42,
                )
                logger.info(
                    f"RandomUnderSampler initialized with sampling_strategy='{sampling_strategy}'",
                )
            case "over":
                self.sampler = RandomOverSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=42,
                )
                logger.info(
                    f"RandomOverSampler initialized with sampling_strategy='{sampling_strategy}'",
                )
            case "smote":
                # Ensure k_neighbors is compatible with the number of samples in minority class
                min_class_size = self.Y.value_counts().min()
                adj_k = min(k_neighbors, min_class_size - 1)
                if adj_k < 1:
                    adj_k = 1
                    logger.warning(
                        f"Class size {min_class_size} is too small. Setting k_neighbors to 1.",
                    )

                if categorical_features is not None and len(categorical_features) > 0:
                    self.sampler = SMOTENC(
                        categorical_features=categorical_features,
                        sampling_strategy=sampling_strategy,
                        random_state=42,
                        k_neighbors=adj_k,
                    )
                    logger.info(
                        f"SMOTENC initialized with categorical_features={categorical_features}, k_neighbors={adj_k} and sampling_strategy='{sampling_strategy}'",
                    )
                else:
                    self.sampler = SMOTE(
                        sampling_strategy=sampling_strategy,
                        random_state=42,
                        k_neighbors=adj_k,
                    )
                    logger.info(
                        f"SMOTE initialized with k_neighbors={adj_k} and sampling_strategy='{sampling_strategy}'",
                    )
            case "adasyn":
                self.sampler = ADASYN(
                    sampling_strategy=sampling_strategy,
                    random_state=42,
                )
                logger.info(
                    f"ADASYN initialized with sampling_strategy='{sampling_strategy}'",
                )
            case _:
                raise ValueError(f"Unsupported sampling method: {method}")

    def resample_data(self, n_target: int = 300, max_iter: int = 20):
        """
        Генерирует уникальные синтетические строки, приводя типы к исходным, удаляя все совпадающие с оригиналом по значениям.
        n_target: сколько уникальных синтетических строк нужно получить (по умолчанию 300)
        max_iter: максимальное число попыток генерации (чтобы не зациклиться)
        """
        if self.sampler is None:
            raise ValueError(
                "Sampler not set. Please call get_sampler() to set the sampling method before plotting.",
            )

        orig_df = pd.concat(
            [self.X.reset_index(drop=True), self.Y.reset_index(drop=True)],
            axis=1,
        )
        orig_df = orig_df.astype(self.data_types.to_dict())
        orig_set = set(tuple(row) for row in orig_df.values)

        unique_synth = []
        synth_columns = list(self.X.columns) + [
            self.Y.name if self.Y.name else "target",
        ]
        n_generated = 0
        iters = 0

        # Определяем категориальные признаки
        categorical_cols = [
            col
            for col in self.X.columns
            if str(self.data_types[col]).startswith("category")
            or str(self.data_types[col]) == "object"
            or (
                str(self.data_types[col]).startswith("int")
                and self.X[col].nunique() <= 4
            )
        ]
        numeric_cols = [col for col in self.X.columns if col not in categorical_cols]

        adasyn_with_cat = self.sampler_method == "adasyn" and len(categorical_cols) > 0
        if adasyn_with_cat:
            logger.warning(
                "ADASYN не поддерживает категориальные признаки. Будет выполнено: 1) генерация только по числовым признакам, 2) заполнение категориальных случайными значениями из оригинального распределения. Итоговые строки могут быть менее реалистичны!",
            )

        while n_generated < n_target and iters < max_iter:
            if adasyn_with_cat:
                # Семплируем только по числовым
                x_num = self.X[numeric_cols]
                min_class_size = self.Y.value_counts().min()
                adj_k = min(5, min_class_size - 1)
                if adj_k < 1:
                    adj_k = 1
                    logger.warning(
                        f"Class size {min_class_size} is too small. Setting k_neighbors to 1.",
                    )
                sampler_num = ADASYN(
                    sampling_strategy="auto",
                    random_state=42,
                    n_neighbors=adj_k,
                )
                x_res, y_res = sampler_num.fit_resample(x_num, self.Y)
                # Собираем датафрейм с числовыми + target
                df_res = pd.concat(
                    [
                        pd.DataFrame(x_res, columns=numeric_cols),
                        pd.Series(y_res, name=self.Y.name),
                    ],
                    axis=1,
                )
                # Добавляем категориальные как NaN
                for col in categorical_cols:
                    df_res[col] = np.nan
                # Приводим к нужному порядку
                df_res = df_res[synth_columns]
                # Заполняем категориальные случайно
                df_res = self._fill_categorical_random(
                    df_res,
                    orig_df,
                    categorical_cols,
                )
                df_res = df_res.astype(self.data_types.to_dict())
            else:
                # Обычный путь (SMOTE/ADASYN без категориальных)
                x_res, y_res = self.sampler.fit_resample(self.X, self.Y)
                df_res = pd.concat(
                    [
                        pd.DataFrame(x_res, columns=self.X.columns),
                        pd.Series(y_res, name=self.Y.name),
                    ],
                    axis=1,
                )
                df_res = df_res.astype(self.data_types.to_dict())
            # Удаляем все строки, совпадающие с оригиналом и уже выбранными синтетическими
            synth_set = set(tuple(row) for row in df_res.values)
            synth_set = synth_set - orig_set
            if unique_synth:
                prev_set = set(tuple(row) for row in unique_synth)
                synth_set = synth_set - prev_set
            for row in synth_set:
                unique_synth.append(row)
                n_generated += 1
                if n_generated >= n_target:
                    break
            iters += 1
            logger.info(
                f"Iteration {iters}: total unique synthetic rows = {n_generated}",
            )
        if n_generated < n_target:
            logger.warning(
                f"Could not generate requested {n_target} unique synthetic rows, got only {n_generated}",
            )
        synth_df = pd.DataFrame(unique_synth, columns=synth_columns)
        self.x = synth_df.drop(columns=[self.Y.name if self.Y.name else "target"])
        self.y = synth_df[self.Y.name if self.Y.name else "target"]
        logger.info(f"Final unique synthetic shape: {self.x.shape}, {self.y.shape}")

    def export_csv(self, path: str, include_index: bool = False, val_round: int = None):
        """
        Export the synthetic dataset to a CSV file.
        Args:
            path: Output file path.
            include_index: Whether to include row indices in the CSV.
            val_round: Number of decimal places to round numeric values.
        """
        if self.x is None or self.y is None:
            raise ValueError("No resampled data available. Call resample_data() first.")

        df = pd.concat(
            [
                pd.DataFrame(self.x),
                pd.Series(
                    self.y,
                    name=self.y.name if self.y.name else "adverse_outcome",
                ),
            ],
            axis=1,
        )
        if val_round is not None:
            float_cols = df.select_dtypes(include=["float"]).columns
            df[float_cols] = df[float_cols].round(val_round)
        df = df.astype(self.data_types.to_dict())
        df.to_csv(path, index=include_index)
        logger.info(
            f"Data exported to '{path}', rows={len(df)}, include_index={include_index}",
        )

    def data_plot(
        self,
        ncols: int = 2,
        figsize: tuple = (12, 6),
        autopct: str = "%1.1f%%",
    ):
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
            axs[0].set_title("Original data")
            y_res.value_counts().plot.pie(autopct=autopct, ax=axs[1])
            axs[1].set_title(f"{self.sampler_method}-sampling")
            return fig, axs
        else:
            raise ValueError(
                "Sampler not set. Please call get_sampler() to set the sampling method before plotting.",
            )

    def synthetic_comparison_plot(self, features=None, figsize=(12, 8)):
        """
        Plot overlay histograms comparing feature distributions between real and synthetic data.
        In 'new' mode: compares original and synthetic data.
        In 'append' mode: compares original and resampled data.
        Args:
            features: List of feature names to compare (default: age, cholesterol_level, syntax_score, angina_functional_class, lumen_area, unstable_plaque).
            figsize: Figure size.
        Returns:
            fig, axs: Matplotlib figure and axes objects.
        """
        if features is None:
            features = [
                "age",
                "cholesterol_level",
                "syntax_score",
                "angina_functional_class",
                "lumen_area",
                "unstable_plaque",
            ]
        n_features = len(features)
        fig, axs = plt.subplots(n_features, 1, figsize=figsize)
        if n_features == 1:
            axs = [axs]
        for i, feat in enumerate(features):
            # Получаем значения для исходных и синтетических
            real_vals = self.X[feat] if hasattr(self.X, "__getitem__") else None
            syn_vals = None
            if self.sampling_mode == "new":
                # Только синтетические
                if hasattr(self.sampler, "sample_indices_"):
                    orig_len = len(self.X)
                    synthetic_mask = [
                        idx for idx in self.sampler.sample_indices_ if idx >= orig_len
                    ]
                    syn_vals = (
                        self.x.iloc[synthetic_mask][feat]
                        if hasattr(self.x, "iloc")
                        else self.x[synthetic_mask, self.x.columns.get_loc(feat)]
                    )
                else:
                    syn_vals = self.x[feat] if hasattr(self.x, "__getitem__") else None
                axs[i].hist(real_vals, bins=20, alpha=0.5, label="Real", color="blue")
                axs[i].hist(
                    syn_vals,
                    bins=20,
                    alpha=0.5,
                    label="Synthetic",
                    color="orange",
                )
                axs[i].set_title(f"{feat}: Real vs Synthetic (new mode)")
            else:
                syn_vals = self.x[feat] if hasattr(self.x, "__getitem__") else None
                axs[i].hist(real_vals, bins=20, alpha=0.5, label="Real", color="blue")
                axs[i].hist(
                    syn_vals,
                    bins=20,
                    alpha=0.5,
                    label="Resampled",
                    color="green",
                )
                axs[i].set_title(f"{feat}: Real vs Resampled (append mode)")
            axs[i].legend()
            axs[i].set_xlabel(feat)
            axs[i].set_ylabel("Count")
        fig.tight_layout()
        return fig, axs


if __name__ == "__main__":
    # Settings for generating 300+ balanced UNIQUE synthetic records
    target_synthetic_total = 300  # Total unique synthetic samples
    sampling_method = "adasyn"  # "smote" or "adasyn"; ADASYN не поддерживает категориальные признаки, используйте только для float/int

    dataloader = MyDataLoader(data_path="backend/data/features.csv")
    dataloader.load_data()
    dataloader.impute_missing_values(n_neighbors=3)

    X, Y = dataloader.get_data_features_and_target(target_column="adverse_outcome")

    # Improved categorical features detection
    categorical_features = []
    if isinstance(X, pd.DataFrame):
        for i, col in enumerate(X.columns):
            dt = X[col].dtype
            nunique = X[col].nunique()
            # Object/Category types or Binary/Ordinal columns with few unique values
            if (
                dt == "object"
                or str(dt).startswith("category")
                or (str(dt).startswith("int") and nunique <= 4)
            ):
                categorical_features.append(i)
                logger.info(
                    f"Feature '{col}' (index {i}) identified as categorical (nunique={nunique})",
                )

    # Для SMOTE: sampling_strategy как dict для балансировки классов
    if sampling_method == "smote":
        n_classes = Y.nunique()
        n_per_class = target_synthetic_total // n_classes
        unique_classes = list(Y.unique())
        sampling_strategy = dict.fromkeys(unique_classes, n_per_class)
    else:
        sampling_strategy = "auto"

    my_data_sampler = DataSampling(
        X=X,
        Y=Y,
        data_types=dataloader.data_types,
        sampling_mode="new",  # ONLY synthetic
    )

    my_data_sampler.get_sampler(
        method=sampling_method,
        sampling_strategy=sampling_strategy,
        categorical_features=categorical_features,
        k_neighbors=5,
    )

    my_data_sampler.resample_data(n_target=target_synthetic_total)

    output_path = f"backend/data/resampled_data_{sampling_method}.csv"
    my_data_sampler.export_csv(path=output_path, include_index=False, val_round=2)

    # Validation and Plotting
    if my_data_sampler.x is not None:
        fig, axs = my_data_sampler.data_plot()
        fig.savefig(f"backend/data/sampling_plot_{sampling_method}.png")
        plt.close(fig)

        # Comparison plot (Real vs Synthetic in 'new' mode)
        fig, axs = my_data_sampler.synthetic_comparison_plot()
        fig.savefig(f"backend/data/synthetic_comparison_{sampling_method}.png")
        plt.close(fig)

        logger.info(f"Process completed. Final synthetic file: {output_path}")
        # Verify the file content
        df_gen = pd.read_csv(output_path)
        logger.info(f"Generated file rows: {len(df_gen)}")
        logger.info(
            f"Generated class distribution:\n{df_gen['adverse_outcome'].value_counts().to_dict()}",
        )

        # Проверка пересечения синтетики и оригинала
        n_inter, inter_rows = my_data_sampler.check_intersection_with_original()
        if n_inter == 0:
            logger.info(
                "No intersection between synthetic and original datasets. All synthetic rows are unique.",
            )
        else:
            logger.warning(
                f"Intersection found! {n_inter} synthetic rows are present in the original dataset. Example(s): {inter_rows}",
            )
