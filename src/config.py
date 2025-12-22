# src/config.py
import os

# === ABSOLUTE PATHS BASED ON YOUR PROJECT LOCATION ===
PROJECT_ROOT = r"C:\Users\h\Desktop\week 5\Fraud-detection"

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")

# Ensure folders exist
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# File names
FRAUD_DATA_FILE = "Fraud_Data.csv"
IP_MAPPING_FILE = "IpAddress_to_Country.csv"
CREDITCARD_FILE = "creditcard.csv"

# Full absolute file paths (for direct use)
FRAUD_DATA_FULL_PATH = os.path.join(RAW_DATA_PATH, FRAUD_DATA_FILE)
IP_MAPPING_FULL_PATH = os.path.join(RAW_DATA_PATH, IP_MAPPING_FILE)
CREDITCARD_FULL_PATH = os.path.join(RAW_DATA_PATH, CREDITCARD_FILE)

# Reproducibility
RANDOM_STATE = 42
SMOTE_RANDOM_STATE = RANDOM_STATE
TEST_SIZE = 0.2

# Feature columns
ECOMMERCE_CATEGORICAL_COLS = ['source', 'browser', 'sex', 'country']
ECOMMERCE_NUMERICAL_COLS = [
    'purchase_value', 'age', 'hour_of_day', 'day_of_week',
    'time_since_signup_hours', 'user_transaction_count', 'device_transaction_count'
]

# Analysis thresholds
MIN_COUNTRY_TXNS_FOR_FRAUD_RATE = 50
MAX_SIGNUP_HOURS_FOR_PLOT = 1000
MAX_DEVICE_COUNT_FOR_PLOT = 20


# Helper function
def ip_to_int(ip_str: str) -> int:
    try:
        parts = ip_str.split('.')
        if len(parts) != 4:
            return float('nan')
        return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
    except:
        return float('nan')

print(f"Config loaded: Project root = {PROJECT_ROOT}")
print(f"Raw data path = {RAW_DATA_PATH}")