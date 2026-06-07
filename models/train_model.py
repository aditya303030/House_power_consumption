from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

features = [
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
]

target = "target_next_hour"

