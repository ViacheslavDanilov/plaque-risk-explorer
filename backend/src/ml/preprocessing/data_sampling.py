import logging
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from backend.src.ml.preprocessing.data_loader import MyDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataSampling:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.x = None
        self.y = None
        self.sampler = None
        self.sampler_method = None

    def get_sampler(self, method: str = "under", sampling_strategy: str = "not minority"):
        self.sampler_method = method
        match method:
            case "under":
                self.sampler = RandomUnderSampler(sampling_strategy=sampling_strategy)
                logger.info(f"RandomUnderSampler initialized with sampling_strategy='{sampling_strategy}'")
            case "over":
                self.sampler = RandomOverSampler(sampling_strategy=sampling_strategy)
                logger.info(f"RandomOverSampler initialized with sampling_strategy='{sampling_strategy}'")

    def resample_data(self):
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

    def export_csv(self, path: str, include_index: bool = False):
        """Save the resampled dataset to a CSV file.

        Combines features and target into a single DataFrame and writes it out.
        """
        if self.x is None or self.y is None:
            raise ValueError("No resampled data available. Call resample_data() first.")

        import pandas as pd

        df = pd.concat([pd.DataFrame(self.x), pd.Series(self.y, name="target")], axis=1)
        df.to_csv(path, index=include_index)
        logger.info(f"Resampled data exported to '{path}', include_index={include_index}")

    def data_plot(self, ncols: int = 2, figsize: tuple = (12, 6), autopct: str = "%1.1f%%"):
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


if __name__ == "__main__":
    # Example usage
    sampling_method = "over"  # "over" or "under"
    sampling_strategy = "auto"  # "not majority", "all", "auto", or a dict specifying the desired number of samples for each class
    dataloader = MyDataLoader(data_path="backend/data/features.csv")
    data = dataloader.load_data()
    X, Y = dataloader.get_data_features_and_target(target_column="adverse_outcome")
    my_data_sampler = DataSampling(
        X = X,
        Y = Y
    )
    my_data_sampler.get_sampler(method=sampling_method, sampling_strategy=sampling_strategy)
    my_data_sampler.resample_data()
    my_data_sampler.export_csv(path=f"backend/data/resampled_data_{my_data_sampler.sampler_method}.csv", include_index=False)
    fig, axs = my_data_sampler.data_plot()
    fig.savefig(f"backend/data/sampling_plot_{my_data_sampler.sampler_method}.png")
    fig.show()
    input("Press Enter to continue...")