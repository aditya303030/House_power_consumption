import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Household Power Forecasting",
    page_icon="⚡",
    layout="wide"
)


# loading model, features, data, and results

@st.cache_resource
def load_model():
    model = joblib.load("models/best_power_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    return model, feature_cols


@st.cache_data
def load_data():
    df = pd.read_parquet("reports/deployment_sample.parquet")
    return df


@st.cache_data
def load_results():
    results = pd.read_csv("reports/model_results.csv")
    return results


@st.cache_data
def load_predictions():
    predictions = pd.read_csv(
        "reports/test_predictions.csv",
        index_col=0,
        parse_dates=True
    )
    return predictions


model, feature_cols = load_model()
df = load_data()
results = load_results()


st.title("⚡ Household Power Consumption Forecasting")

st.write(
    "This app uses a machine learning model to predict household power consumption "
    "one hour into the future using historical smart-meter data."
)


# dataset overview

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Deployment sample rows", f"{df.shape[0]:,}")

with col2:
    st.metric("Features used", len(feature_cols))

with col3:
    st.metric("Model", type(model).__name__)


if isinstance(df.index, pd.DatetimeIndex):
    st.write(f"Date range: `{df.index.min()}` to `{df.index.max()}`")


# model performance

st.subheader("Model Performance")

st.dataframe(results)

best_model = results.sort_values("MAE").iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Best MAE", round(best_model["MAE"], 4))

with col2:
    st.metric("Best RMSE", round(best_model["RMSE"], 4))

with col3:
    st.metric("Best R²", round(best_model["R^2"], 4))


# latest prediction

st.subheader("Next-Hour Forecast")

latest_features = df[feature_cols].iloc[[-1]]
prediction = model.predict(latest_features)[0]

st.metric(
    "Predicted Global Active Power 1 Hour Ahead",
    round(prediction, 4)
)

if isinstance(df.index, pd.DatetimeIndex):
    st.write(f"Latest timestamp used: `{df.index[-1]}`")


with st.expander("See latest feature values"):
    st.dataframe(latest_features)


# actual vs predicted chart

st.subheader("Actual vs Predicted Power Usage")

try:
    predictions = load_predictions()

    # use only a sample so the app stays fast
    sample_preds = predictions.tail(1000)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(sample_preds.index, sample_preds["actual"], label="Actual")
    ax.plot(
        sample_preds.index,
        sample_preds["hist_gradient_boosting"],
        label="HistGradientBoosting Prediction"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Global Active Power")
    ax.set_title("Actual vs Predicted Power Consumption")
    ax.legend()

    st.pyplot(fig)

except Exception as e:
    st.warning(f"Could not load prediction chart: {e}")


# features

st.subheader("Features Used by the Model")

st.write(feature_cols)