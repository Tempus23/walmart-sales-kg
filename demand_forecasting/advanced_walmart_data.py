"""
Advanced Walmart Data Loader - Replicated from Parth Patel's approach.

This module downloads and processes the COMPLETE Walmart dataset from Kaggle,
including stores.csv, features.csv, and train.csv to match the competition setup.
"""

from pathlib import Path
import kaggle
import pandas as pd
import numpy as np
from enum import Enum
from datetime import datetime


# Dataset completo de la competición original
DATASET_SLUG = "mikhail1681/walmart-sales"
DATA_PATH = Path("data/")


class WalmartFeaturesAdvanced(str, Enum):
    """Enhanced feature set matching Parth Patel's implementation."""
    
    # Original dataset features
    STORE = "Store"
    DEPT = "Dept"
    DATE = "Date"
    WEEKLY_SALES = "Weekly_Sales"
    IS_HOLIDAY = "IsHoliday"
    
    # From stores.csv
    TYPE = "Type"
    SIZE = "Size"
    
    # From features.csv
    TEMPERATURE = "Temperature"
    FUEL_PRICE = "Fuel_Price"
    MARKDOWN1 = "MarkDown1"
    MARKDOWN2 = "MarkDown2"
    MARKDOWN3 = "MarkDown3"
    MARKDOWN4 = "MarkDown4"
    MARKDOWN5 = "MarkDown5"
    CPI = "CPI"
    UNEMPLOYMENT = "Unemployment"
    
    # Temporal features (engineered)
    YEAR = "Year"
    MONTH = "Month"
    WEEK = "Week"
    WEEK_OF_YEAR = "WeekOfYear"
    
    # Individual holiday flags (CRITICAL for Parth's performance)
    SUPER_BOWL = "Super_Bowl"
    LABOR_DAY = "Labor_Day"
    THANKSGIVING = "Thanksgiving"
    CHRISTMAS = "Christmas"
    
    # Cyclical encodings
    MONTH_SIN = "MonthSin"
    MONTH_COS = "MonthCos"
    WEEK_SIN = "WeekSin"
    WEEK_COS = "WeekCos"
    
    # Store-specific lags and rolling means
    SALES_LAG_1 = "sales_lag_1"
    SALES_LAG_4 = "sales_lag_4"
    SALES_LAG_52 = "sales_lag_52"
    
    SALES_ROLL_MEAN_4 = "sales_roll_mean_4"
    SALES_ROLL_MEAN_12 = "sales_roll_mean_12"
    SALES_ROLL_STD_4 = "sales_roll_std_4"
    
    # Department-specific features
    DEPT_MEAN_SALES = "dept_mean_sales"
    STORE_MEAN_SALES = "store_mean_sales"


class AdvancedWalmartDataloader:
    """
    Complete dataloader matching the Kaggle competition structure.
    Merges stores.csv, features.csv, and train.csv like Parth Patel's approach.
    """
    
    def __init__(self, dataset_slug: str = DATASET_SLUG, data_path: Path = DATA_PATH):
        self.dataset_slug = dataset_slug
        self.data_path = data_path
        
        # Major holiday dates (from Walmart competition)
        self.holiday_dates = {
            'Super_Bowl': [
                '2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08'
            ],
            'Labor_Day': [
                '2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06'
            ],
            'Thanksgiving': [
                '2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29'
            ],
            'Christmas': [
                '2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27'
            ]
        }
    
    def download_kaggle_dataset(self) -> None:
        """Downloads the complete Walmart dataset from Kaggle."""
        # Check if data already exists
        csv_files = list(self.data_path.glob("*.csv"))
        if len(csv_files) >= 3:  # Need stores, features, train
            print(f"✓ Dataset files found in '{self.data_path}/'. Skipping download.")
            return
        
        print(f"Downloading Walmart dataset from Kaggle: '{self.dataset_slug}'...")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        try:
            kaggle.api.dataset_download_files(
                self.dataset_slug, 
                path=str(self.data_path), 
                unzip=True, 
                quiet=False
            )
            print(f"✓ Data downloaded successfully to '{self.data_path}/'")
        except Exception as e:
            print(f"✗ Error downloading data: {e}")
            raise
    
    def load_raw_data(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        """
        Loads the three main CSV files.
        
        Returns:
            tuple: (stores_df, features_df, train_df) - any can be None if not found
        """
        self.download_kaggle_dataset()
        
        # Find the CSV files
        csv_files = {f.stem: f for f in self.data_path.glob("*.csv")}
        
        print(f"\n📁 Found CSV files: {list(csv_files.keys())}")
        
        # Try different naming conventions
        stores_df = None
        features_df = None
        train_df = None
        
        for name, path in csv_files.items():
            name_lower = name.lower()
            if 'store' in name_lower and 'sales' not in name_lower:
                stores_df = pd.read_csv(path)
                print(f"✓ Loaded stores: {path.name} ({len(stores_df)} rows)")
            elif 'feature' in name_lower:
                features_df = pd.read_csv(path)
                print(f"✓ Loaded features: {path.name} ({len(features_df)} rows)")
            elif 'train' in name_lower or 'sales' in name_lower:
                train_df = pd.read_csv(path)
                print(f"✓ Loaded train/sales: {path.name} ({len(train_df)} rows)")
        
        # If we only have one file (backward compatibility)
        if train_df is None and len(csv_files) == 1:
            single_file = list(csv_files.values())[0]
            train_df = pd.read_csv(single_file)
            print(f"⚠ Single file mode: {single_file.name} ({len(train_df)} rows)")
            print("Note: Missing stores.csv and features.csv - performance will be limited!")
        
        return stores_df, features_df, train_df
    
    def merge_datasets(self, stores_df, features_df, train_df) -> pd.DataFrame:
        """
        Merges the three dataframes like in the Kaggle competition.
        
        Merging strategy (following Parth Patel):
        1. Merge train + stores on 'Store'
        2. Merge result + features on ['Store', 'Date', 'IsHoliday']
        """
        print("\n🔗 Merging datasets...")
        
        # If we don't have separate files, return train as-is
        if stores_df is None and features_df is None:
            print("⚠ No stores/features files - using single dataset")
            return train_df
        
        df = train_df.copy()
        
        # Merge with stores (add Type and Size)
        if stores_df is not None:
            df = df.merge(stores_df, on='Store', how='left')
            print(f"✓ Merged with stores: {df.shape}")
        
        # Merge with features (add economic indicators and markdowns)
        if features_df is not None:
            # Ensure Date format matches
            if 'Date' in features_df.columns:
                df = df.merge(features_df, on=['Store', 'Date', 'IsHoliday'], how='left')
                print(f"✓ Merged with features: {df.shape}")
        
        print(f"✓ Final merged dataset: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        return df
    
    def add_holiday_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds individual holiday dummy variables (CRITICAL feature).
        
        This is what Parth Patel emphasizes - each holiday has different impact!
        """
        print("\n🎄 Adding individual holiday flags...")
        
        # Convert Date to datetime if needed
        if df['Date'].dtype == 'object':
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Initialize all holiday columns to False
        for holiday in self.holiday_dates.keys():
            df[holiday] = False
        
        # Mark specific holiday weeks
        for holiday, dates in self.holiday_dates.items():
            holiday_dates_dt = pd.to_datetime(dates)
            df[holiday] = df['Date'].isin(holiday_dates_dt)
            count = df[holiday].sum()
            print(f"  ✓ {holiday}: {count} weeks marked")
        
        return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds time-based features (Year, Month, Week, etc.)."""
        print("\n📅 Generating temporal features...")
        
        if df['Date'].dtype == 'object':
            df['Date'] = pd.to_datetime(df['Date'])
        
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
        df['WeekOfYear'] = df['Week']  # Alias
        
        # Cyclical encoding (important for capturing seasonality)
        df['MonthSin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['MonthCos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['WeekSin'] = np.sin(2 * np.pi * df['Week'] / 52)
        df['WeekCos'] = np.cos(2 * np.pi * df['Week'] / 52)
        
        print(f"✓ Added: Year, Month, Week, Cyclical encodings")
        
        return df
    
    def add_lag_and_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds lagged sales and rolling statistics.
        GROUPED BY STORE-DEPT to avoid data leakage!
        """
        print("\n📊 Generating lag & rolling features...")
        
        # Sort by Store, Dept, Date to ensure proper ordering
        df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
        
        # Group by Store and Dept (critical!)
        grouped = df.groupby(['Store', 'Dept'])
        
        # Lags (previous weeks)
        df['sales_lag_1'] = grouped['Weekly_Sales'].shift(1)
        df['sales_lag_4'] = grouped['Weekly_Sales'].shift(4)  # 1 month ago
        df['sales_lag_52'] = grouped['Weekly_Sales'].shift(52)  # 1 year ago
        
        # Rolling statistics (exclude current week!)
        df['sales_roll_mean_4'] = grouped['Weekly_Sales'].shift(1).rolling(window=4, min_periods=1).mean()
        df['sales_roll_mean_12'] = grouped['Weekly_Sales'].shift(1).rolling(window=12, min_periods=1).mean()
        df['sales_roll_std_4'] = grouped['Weekly_Sales'].shift(1).rolling(window=4, min_periods=1).std()
        
        print(f"✓ Added: Lags (1, 4, 52 weeks), Rolling mean/std")
        
        return df
    
    def add_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds department and store-level aggregated statistics."""
        print("\n🏬 Generating aggregated features...")
        
        # Department average sales (across all stores)
        dept_means = df.groupby('Dept')['Weekly_Sales'].mean()
        df['dept_mean_sales'] = df['Dept'].map(dept_means)
        
        # Store average sales (across all departments)
        store_means = df.groupby('Store')['Weekly_Sales'].mean()
        df['store_mean_sales'] = df['Store'].map(store_means)
        
        print(f"✓ Added: Dept & Store mean sales")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the merged dataset following Parth Patel's approach.
        
        Steps:
        1. Fill MarkDown NAs with 0 (no promotion)
        2. Remove negative sales (returns/anomalies)
        3. Handle remaining NAs from lags (fill with 0 or drop)
        """
        print("\n🧹 Cleaning data...")
        
        initial_rows = len(df)
        
        # 1. Fill MarkDown columns with 0 (assumption: no promotion if missing)
        markdown_cols = [c for c in df.columns if 'MarkDown' in c]
        if markdown_cols:
            df[markdown_cols] = df[markdown_cols].fillna(0)
            print(f"✓ Filled {len(markdown_cols)} MarkDown columns with 0")
        
        # 2. Remove negative sales (outliers from returns)
        negative_mask = df['Weekly_Sales'] <= 0
        negative_count = negative_mask.sum()
        if negative_count > 0:
            df = df[~negative_mask].reset_index(drop=True)
            print(f"✓ Removed {negative_count} rows with negative sales")
        
        # 3. Fill lag/rolling NAs with 0 (for first weeks)
        lag_cols = [c for c in df.columns if 'lag' in c or 'roll' in c]
        if lag_cols:
            df[lag_cols] = df[lag_cols].fillna(0)
            print(f"✓ Filled {len(lag_cols)} lag/rolling columns with 0")
        
        # 4. Drop any remaining NAs in critical columns
        critical_cols = ['Weekly_Sales', 'Store', 'Date']
        if 'Dept' in df.columns:
            critical_cols.append('Dept')
        
        df = df.dropna(subset=critical_cols)
        
        final_rows = len(df)
        removed = initial_rows - final_rows
        
        print(f"✓ Final dataset: {final_rows} rows ({removed} removed)")
        print(f"✓ Remaining NAs: {df.isna().sum().sum()}")
        
        return df
    
    def get_complete_data(self) -> pd.DataFrame:
        """
        Main method: Loads, merges, engineers features, and cleans data.
        
        Returns:
            Complete DataFrame ready for modeling, matching Parth Patel's setup.
        """
        print("\n" + "="*60)
        print("🚀 ADVANCED WALMART DATA PIPELINE - PARTH PATEL REPLICATION")
        print("="*60)
        
        # Load raw data
        stores_df, features_df, train_df = self.load_raw_data()
        
        # Merge datasets
        df = self.merge_datasets(stores_df, features_df, train_df)
        
        # Feature engineering pipeline
        df = self.add_temporal_features(df)
        df = self.add_holiday_flags(df)
        
        # Only add Store-Dept features if we have Dept column
        if 'Dept' in df.columns:
            df = self.add_lag_and_rolling_features(df)
            df = self.add_aggregated_features(df)
        
        # Clean final dataset
        df = self.clean_data(df)
        
        print("\n" + "="*60)
        print(f"✅ PIPELINE COMPLETE: {df.shape}")
        print("="*60)
        print(f"\nFeatures available: {list(df.columns)}\n")
        
        return df


def calculate_wmae(y_true, y_pred, is_holiday, verbose=True):
    """
    Calculates Weighted Mean Absolute Error (WMAE) - Kaggle competition metric.
    
    Holiday weeks are weighted 5x more than regular weeks.
    
    Args:
        y_true: Actual sales
        y_pred: Predicted sales
        is_holiday: Boolean series indicating holiday weeks
        verbose: Print details
        
    Returns:
        WMAE score
    """
    weights = is_holiday.apply(lambda x: 5 if x else 1)
    error_absolute = np.abs(y_true - y_pred)
    wmae = (weights * error_absolute).sum() / weights.sum()
    
    if verbose:
        mae_regular = error_absolute[~is_holiday].mean()
        mae_holiday = error_absolute[is_holiday].mean()
        
        print(f"\n📊 WMAE BREAKDOWN:")
        print(f"  Overall WMAE: ${wmae:,.2f}")
        print(f"  Regular weeks MAE: ${mae_regular:,.2f}")
        print(f"  Holiday weeks MAE: ${mae_holiday:,.2f}")
        print(f"  Holiday weight: 5x")
    
    return wmae
