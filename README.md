# Fraud Detection Project

## Overview

This repository contains a complete **Fraud Detection pipeline** for e-commerce and credit card datasets.  
The project focuses on **Task 1: Data Analysis and Preprocessing**, preparing clean, feature-rich datasets ready for modeling.

The preprocessing pipeline includes:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Geolocation Integration
- Feature Engineering
- Data Transformation
- Handling Class Imbalance

---

## Datasets

The project uses the following datasets (place these in the `data/` folder):

1. **Fraud_Data.csv** – e-commerce transaction data
2. **IpAddress_to_Country.csv** – mapping IP ranges to countries
3. **creditcard.csv** – credit card transaction dataset

---

## Features Engineered

### Time-Based Features

- `hour_of_day` – hour when the purchase occurred
- `day_of_week` – day of the week
- `time_since_signup_hours` – duration between signup and purchase

### Transaction Frequency Features

- `user_transaction_count` – number of transactions per user
- `device_transaction_count` – number of transactions per device

---

## Preprocessing Steps

1. **Data Cleaning**

   - Handle missing values (drop or impute with justification)
   - Remove duplicate records
   - Correct data types (`signup_time`, `purchase_time` → datetime)

2. **Exploratory Data Analysis (EDA)**

   - Univariate analysis of key variables
   - Bivariate analysis (features vs target)
   - Class distribution and imbalance visualization

3. **Geolocation Integration**

   - Convert IP addresses to integer format
   - Merge `Fraud_Data.csv` with `IpAddress_to_Country.csv`
   - Analyze fraud patterns by country

4. **Feature Engineering**

   - Time-based features
   - Transaction frequency/velocity features

5. **Data Transformation**

   - Scale numerical features (`StandardScaler`)
   - Encode categorical features (`OneHotEncoder`)

6. **Handling Class Imbalance**
   - Apply **SMOTE** on training data only
   - Document class distribution before and after resampling

---

## Project Structure

Fraud-detection/
│
├─ data/ # Raw datasets
├─ src/ # Source code
│ ├─ data_preprocessor.py
│ ├─ feature_engineering.py
│ └─ config.py
├─ notebooks/ # Jupyter notebooks for EDA and experiments
├─ processed/ # Processed datasets (pickled features/targets)
├─ requirements.txt
└─ README.md

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/nifomohad/fraud-detection.git
cd Fraud-detection
```

2. Create and activate a virtual environment:

python -m venv .venv

# Windows

.venv\Scripts\activate

# macOS/Linux

source .venv/bin/activate

3. Install dependencies:

pip install -r requirements.txt

Usage

Run the preprocessing pipeline:

python src/data_preprocessor.py

This will:

Clean datasets

Merge geolocation data

Engineer features

Scale and encode data

Apply SMOTE for class imbalance

Save processed datasets in the processed/ folder
