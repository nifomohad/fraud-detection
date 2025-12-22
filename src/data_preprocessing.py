# src/data_preprocessor.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    FRAUD_DATA_FULL_PATH,
    IP_MAPPING_FULL_PATH,
    CREDITCARD_FULL_PATH,
    RANDOM_STATE,
    ECOMMERCE_CATEGORICAL_COLS,
    ECOMMERCE_NUMERICAL_COLS,
    ip_to_int,
    MIN_COUNTRY_TXNS_FOR_FRAUD_RATE
)


class FraudDataPreprocessor:
    """
    Complete preprocessing pipeline for both e-commerce (Fraud_Data.csv) and credit card datasets.
    Handles cleaning, geolocation merge, feature engineering, analysis, and saving processed data.
    """

    def __init__(self):
        self.raw_path = RAW_DATA_PATH
        self.processed_path = PROCESSED_DATA_PATH
        os.makedirs(self.processed_path, exist_ok=True)
        self.preprocessor = None  # Will hold the ColumnTransformer for e-commerce features
        self.fraud_df = None
        self.ip_df = None
        self.credit_df = None

    def load_data(self):
        """Load all three raw datasets using absolute paths from config."""
        print("Loading datasets using absolute paths...\n")
        try:
            self.fraud_df = pd.read_csv(FRAUD_DATA_FULL_PATH)
            self.ip_df = pd.read_csv(IP_MAPPING_FULL_PATH)
            self.credit_df = pd.read_csv(CREDITCARD_FULL_PATH)

            print(f"✅ Fraud_Data loaded: {self.fraud_df.shape}")
            print(f"✅ IP Mapping loaded: {self.ip_df.shape}")
            print(f"✅ Credit Card loaded: {self.credit_df.shape}\n")
        except Exception as e:
            print(f" Error loading data: {e}")
            raise

    def clean_fraud_data(self):
        """Clean the e-commerce fraud dataset: data types, duplicates, missing."""
        print("Cleaning Fraud_Data...\n")
        print(f"Missing values before cleaning:\n{self.fraud_df.isnull().sum()}\n")
        print(f"Duplicate rows before cleaning: {self.fraud_df.duplicated().sum()}\n")

        # Convert timestamps
        self.fraud_df['signup_time'] = pd.to_datetime(self.fraud_df['signup_time'])
        self.fraud_df['purchase_time'] = pd.to_datetime(self.fraud_df['purchase_time'])

        # Convert IP to integer for geolocation merge
        self.fraud_df['ip_address_int'] = self.fraud_df['ip_address'].apply(ip_to_int)

        # Drop duplicates
        self.fraud_df.drop_duplicates(inplace=True)

        print(f"Duplicate rows after cleaning: {self.fraud_df.duplicated().sum()}")
        print("Cleaning complete.\n")

    def merge_geolocation(self):
        """Merge IP ranges to get country for each transaction."""
        print("Merging geolocation data (IP → Country)...\n")
        ip_sorted = self.ip_df.sort_values('lower_bound_ip_address').reset_index(drop=True)
        bounds = ip_sorted['lower_bound_ip_address'].values

        # Efficient range lookup
        indices = np.searchsorted(bounds, self.fraud_df['ip_address_int'].values) - 1
        indices = np.clip(indices, 0, len(ip_sorted) - 1)

        self.fraud_df['country'] = ip_sorted.iloc[indices]['country'].values
        self.fraud_df['country'].fillna('Unknown', inplace=True)

        print(f"Unique countries mapped: {self.fraud_df['country'].nunique()}")
        print("Geolocation merge complete.\n")

    def feature_engineering(self):
        """Create new meaningful features from raw data."""
        print("Feature Engineering...\n")
        df = self.fraud_df

        # Time-based features
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['time_since_signup_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600.0

        # Transaction velocity / frequency
        df['user_transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
        df['device_transaction_count'] = df.groupby('device_id')['device_id'].transform('count')

        print("New features created:")
        print("   - hour_of_day, day_of_week")
        print("   - time_since_signup_hours")
        print("   - user_transaction_count, device_transaction_count\n")

    def analyze_fraud_by_country(self):
        """Print top countries by fraud rate (with minimum transaction threshold)."""
        print(f"Top 10 countries by fraud rate (min {MIN_COUNTRY_TXNS_FOR_FRAUD_RATE} transactions):\n")
        country_stats = (
            self.fraud_df.groupby('country')['class']
            .agg(['mean', 'count'])
            .rename(columns={'mean': 'fraud_rate'})
        )
        filtered = country_stats[country_stats['count'] >= MIN_COUNTRY_TXNS_FOR_FRAUD_RATE]
        top_risky = filtered.sort_values('fraud_rate', ascending=False).head(10)
        print(top_risky.round(4))
        print()

    def preprocess_for_modeling(self):
        """Scale numerical and encode categorical features for e-commerce data."""
        print("Preprocessing e-commerce features for modeling...\n")
        feature_cols = ECOMMERCE_CATEGORICAL_COLS + ECOMMERCE_NUMERICAL_COLS
        X = self.fraud_df[feature_cols]
        y = self.fraud_df['class']

        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), ECOMMERCE_NUMERICAL_COLS),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ECOMMERCE_CATEGORICAL_COLS)
        ])

        X_transformed = self.preprocessor.fit_transform(X)
        print(f"Transformed feature matrix shape: {X_transformed.shape}")

        # Save processed data
        features_path = os.path.join(self.processed_path, "ecommerce_features.pkl")
        target_path = os.path.join(self.processed_path, "ecommerce_target.pkl")

        pd.DataFrame(X_transformed).to_pickle(features_path)
        pd.Series(y).to_pickle(target_path)

        print(f"E-commerce features saved to: {features_path}")
        print(f"E-commerce target saved to: {target_path}\n")

    def process_creditcard(self):
        """Simple preprocessing for creditcard.csv: deduplicate and scale."""
        print("Processing creditcard.csv...\n")
        df = self.credit_df.copy()
        print(f"Duplicates before removal: {df.duplicated().sum()}")
        df.drop_duplicates(inplace=True)
        print(f"Shape after deduplication: {df.shape}")

        X = df.drop('Class', axis=1)
        y = df['Class']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        features_path = os.path.join(self.processed_path, "creditcard_features.pkl")
        target_path = os.path.join(self.processed_path, "creditcard_target.pkl")

        pd.DataFrame(X_scaled, columns=X.columns).to_pickle(features_path)
        pd.Series(y).to_pickle(target_path)

        print(f"Credit card features saved to: {features_path}")
        print(f"Credit card target saved to: {target_path}\n")

    def handle_imbalance_smote(self, X, y):
        """Apply SMOTE only on training data (example method)."""
        print("Applying SMOTE for class imbalance...")
        print(f"Before SMOTE: {np.bincount(y)}")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X, y)
        print(f"After SMOTE: {np.bincount(y_res)}")
        return X_res, y_res

    def run_full_pipeline(self):
        """Execute the complete Task 1 pipeline."""
        print("Starting Full Fraud Detection Preprocessing Pipeline\n" + "="*60)
        self.load_data()
        self.clean_fraud_data()
        self.merge_geolocation()
        self.feature_engineering()
        self.analyze_fraud_by_country()
        self.preprocess_for_modeling()
        self.process_creditcard()
        print("="*60)
        print("Task 1 Pipeline Completed Successfully!")
        print("All processed files are saved in:", self.processed_path)


# For direct script execution
if __name__ == "__main__":
    preprocessor = FraudDataPreprocessor()
    preprocessor.run_full_pipeline()