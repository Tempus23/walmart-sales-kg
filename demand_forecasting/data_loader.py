"""
This script downloads the Walmart Sales dataset from Kaggle.

It requires the Kaggle API to be configured with your credentials.
See: https://www.kaggle.com/docs/api
"""
import os
from pathlib import Path
import kaggle
import pandas as pd


DATASET_SLUG = "mikhail1681/walmart-sales"
DATA_PATH = Path("data/")


def download_kaggle_dataset(dataset_slug: str, path: Path) -> None:
    """
    Downloads and unzips dataset files from Kaggle to the specified path.

    Args:
        dataset_slug (str): The slug of the Kaggle dataset (e.g., 'user/dataset-name').
        path (Path): The directory to download the files into.
    """
    if any(path.iterdir()):
        print(f"There are already files in '{path}/'. Skipping download.")
        return
    
    print(f"Downloading data for Kaggle dataset: '{dataset_slug}'...")
    
    path.mkdir(parents=True, exist_ok=True)

    try:
        kaggle.api.dataset_download_files(dataset_slug, path=str(path), unzip=True, quiet=False)
        print(f"Data downloaded and unzipped successfully to '{path}/'")
    except Exception as e:
        print(f"An error occurred while downloading the data: {e}")

def get_data(path = DATA_PATH) -> pd.DataFrame:
    """Loads the main sales data into a DataFrame."""
    data_files = [f for f in path.iterdir() if f.is_file() and f.suffix == '.csv']

    if len(data_files) != 1:
        raise FileNotFoundError(f"Expected exactly one CSV file in the directory: {path}, found {len(data_files)}")
    
    df = pd.read_csv(data_files[0])
    return df

if __name__ == "__main__":
    get_data()