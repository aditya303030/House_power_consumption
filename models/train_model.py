import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_parquet('power_clean.parquet')

# print("Loaded data:")
# print(df.head)

if not isinstance(df.index, pd.DatetimeIndex):
    raise TypeError("Expected df index to be a DatetimeIndex. Check your saved parquet file.")


feature_cols = [
    "Global_active_power", 
    "hour",
    "day_of_week",
    "month",
    "is_weekend",

    "power_lag_1",
    "power_lag_60",
    "power_lag_1440",

    "rolling_mean_60",
    "rolling_std_60",
    "rolling_mean_1440",

    "Global_reactive_power",
    "Voltage",
    "Global_intensity",

    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",

    "total_metered",
    "unmetered_energy",
    "global_active_energy"
]

# validating if all columns exist
target_col = "target_next_hour"

required_cols = feature_cols + [target_col]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"These required columns are missing from the dataframe: {missing_cols}")


# creating a subset of df called model_df with features and target columns
model_df = df[required_cols]

# print('dataframe for model')
# print(model_df)

# missing values validation check
missing_count = model_df.isna().sum().sum()

if missing_count > 0:
    print(model_df.isna().sum())
    raise ValueError("There are still missing values in the dataset")

X = model_df[feature_cols]
y = model_df[target_col]

# because this is a time-series forcasting project, we have to create the train and test split manually instead of train_test_split()

split_index = int(len(model_df)*0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)