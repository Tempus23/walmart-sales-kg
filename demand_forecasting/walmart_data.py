"""
This script downloads the Walmart Sales dataset from Kaggle.

It requires the Kaggle API to be configured with your credentials.
See: https://www.kaggle.com/docs/api
"""

from pathlib import Path
import kaggle
import pandas as pd
from enum import Enum


DATASET_SLUG = "mikhail1681/walmart-sales"
DATA_PATH = Path("data/")
MAX_PRINTS = 10
ACTUAL_PRINTS = 0


class WalmartFeatures(str, Enum):
    STORE = "Store"
    DATE = "Date"
    WEEKLY_SALES = "Weekly_Sales"
    HOLIDAY_FLAG = "Holiday_Flag"
    TEMPERATURE = "Temperature"
    FUEL_PRICE = "Fuel_Price"
    CPI = "CPI"
    UNEMPLOYMENT = "Unemployment"

    # Feature engineering
    MONTH = "Month"
    QUARTER = "Quarter"
    VENTAS_LAG_1 = "ventas_lag_1"
    VENTAS_LAG_4 = "ventas_lag_4"
    VENTAS_LAG_52 = "ventas_lag_52"
    MEDIA_MOVIL_4_SEMANAS = "media_movil_4_semanas"
    MEDIA_MOVIL_12_SEMANAS = "media_movil_12_semanas"


class WalmartDataloader:
    def __init__(self, dataset_slug: str = DATASET_SLUG, data_path: Path = DATA_PATH):
        self.dataset_slug = dataset_slug
        self.data_path = data_path

    def download_kaggle_dataset(self) -> None:
        """
        Downloads and unzips dataset files from Kaggle to the specified path.

        Args:
            dataset_slug (str): The slug of the Kaggle dataset (e.g., 'user/dataset-name').
            path (Path): The directory to download the files into.
        """
        if any(self.data_path.iterdir()):
            print(f"There are already files in '{self.data_path}/'. Skipping download.")
            return

        print(f"Downloading data for Kaggle dataset: '{self.dataset_slug}'...")

        self.data_path.mkdir(parents=True, exist_ok=True)

        try:
            kaggle.api.dataset_download_files(
                self.dataset_slug, path=str(self.data_path), unzip=True, quiet=False
            )
            print(f"Data downloaded and unzipped successfully to '{self.data_path}/'")
        except Exception as e:
            print(f"An error occurred while downloading the data: {e}")

    def get_data(self) -> pd.DataFrame:
        """Loads the main sales data into a DataFrame."""
        data_files = [
            f for f in self.data_path.iterdir() if f.is_file() and f.suffix == ".csv"
        ]

        if len(data_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one CSV file in the directory: {self.data_path}, found {len(data_files)}"
            )

        df = pd.read_csv(data_files[0])

        return df

    def _clean_na_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the DataFrame by handling missing values."""
        # Example cleaning: Drop rows with any missing values
        df = df.dropna().reset_index(drop=True)
        return df

    def _date_parsing(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        """Parses date columns to datetime format."""
        if key not in df.columns:
            raise KeyError(f"Column '{key}' not found in DataFrame.")

        df[key] = pd.to_datetime(
            df[key], format="%d-%m-%Y", dayfirst=True, errors="raise"
        )
        return df

    def get_clean_data(self) -> pd.DataFrame:
        """Loads, parses, cleans, and augments the sales data."""
        df = self.get_data()
        df = self._date_parsing(df, WalmartFeatures.DATE)
        df = self._clean_na_values(df)
        df = self.add_features_to_data(df)
        return df

    @staticmethod
    def add_features_to_data(df: pd.DataFrame) -> pd.DataFrame:

        df[WalmartFeatures.MONTH] = df[WalmartFeatures.DATE].dt.month
        df[WalmartFeatures.QUARTER] = df[WalmartFeatures.DATE].dt.quarter

        agrupado = df.groupby([WalmartFeatures.STORE])

        df[WalmartFeatures.VENTAS_LAG_1] = agrupado[WalmartFeatures.WEEKLY_SALES].shift(
            1
        )
        df[WalmartFeatures.VENTAS_LAG_4] = agrupado[WalmartFeatures.WEEKLY_SALES].shift(
            4
        )  # 1 mes
        df[WalmartFeatures.VENTAS_LAG_52] = agrupado[
            WalmartFeatures.WEEKLY_SALES
        ].shift(
            52
        )  # 1 año

        df[WalmartFeatures.MEDIA_MOVIL_4_SEMANAS] = (
            agrupado[WalmartFeatures.WEEKLY_SALES]
            .shift(1)
            .rolling(window=4, min_periods=1)
            .mean()
        )  # 1 mes
        df[WalmartFeatures.MEDIA_MOVIL_12_SEMANAS] = (
            agrupado[WalmartFeatures.WEEKLY_SALES]
            .shift(1)
            .rolling(window=12, min_periods=1)
            .mean()
        )  # 1 año

        df = df.fillna(0)

        return df


if __name__ == "__main__":
    data_loader = WalmartDataloader()
    df = data_loader.get_clean_data()

    # plt.figure(figsize=(8,5))
    # sns.heatmap(df.corr(), annot=True, cmap="Purples")
    # plt.title("Correlation Heatmap")
    # plt.show()
