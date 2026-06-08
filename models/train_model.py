import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
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
# time based train/test split

split_index = int(len(model_df)*0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)


# Model evaluation function

def evaluate_model(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name}")
    print("Mean absolute error (MAE) :", mae)
    print("Root mean squared error (RMSE) :", rmse)
    print("R^2 :", r2)

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R^2": r2
    }

results = []

# Baseline model 1 - prediction for power usage for 1 hour from now is the same as the current power usage

baseline_current_pred = X_test["Global_active_power"]
results.append(evaluate_model("baseline_current_power", y_test, baseline_current_pred))


# Baseline model 2 - prediction for power usage for 1 hour from now is the same as the power usage 24 hours ago

baseline_yesterday_pred = X_test["power_lag_1440"]
results.append(evaluate_model("baseline_yesterday_power", y_test, baseline_yesterday_pred))

# linear regression model

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

results.append(evaluate_model("Linear Regression Model", y_test, linear_pred))

# HistGradientBoostingRegressor

hgb_model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42)
hgb_model.fit(X_train, y_train)

hgb_pred = hgb_model.predict(X_test)
results.append(evaluate_model("HistGradientBoostingModel", y_test, hgb_pred))

# Random Forest Model

rf_model = RandomForestRegressor(n_estimators=50, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

results.append(evaluate_model("Random Forest", y_test, rf_pred))


# comparing all the models

results_df = pd.DataFrame(results).sort_values("MAE")

print("\nModel comparison:")
print(results_df)