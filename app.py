import streamlit as st
import pandas as pd
import joblib
import os
import sys

st.set_page_config(
    page_title="F1 Pit Stop Prediction",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ F1 Pit Stop Prediction - Cloud Diagnostic Test")

st.write("Python version:")
st.code(sys.version)

st.write("Current working directory:")
st.code(os.getcwd())

st.write("Main folder files:")
st.write(os.listdir("."))

if os.path.exists("model"):
    st.write("Model folder files:")
    st.write(os.listdir("model"))
else:
    st.error("Model folder is missing.")

MODEL_PATH = "model/pitstop_model.pkl"
COLUMNS_PATH = "model/model_columns.pkl"


def prepare_input(input_df, model_columns):
    input_df = input_df.copy()
    input_df.columns = input_df.columns.str.strip()

    if "PitNextLap" in input_df.columns:
        input_df = input_df.drop(columns=["PitNextLap"])

    if "Compound" in input_df.columns:
        input_encoded = pd.get_dummies(input_df, columns=["Compound"], drop_first=True)
    else:
        input_encoded = input_df.copy()

    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    return input_encoded


st.header("Manual Input")

col1, col2, col3 = st.columns(3)

with col1:
    stint = st.number_input("Stint", min_value=1, max_value=10, value=2)
    tyre_life = st.number_input("Tyre Life", min_value=0, max_value=80, value=28)
    position = st.number_input("Position", min_value=1, max_value=20, value=10)

with col2:
    lap_time = st.number_input("Lap Time (s)", min_value=40.0, max_value=200.0, value=100.0)
    lap_time_delta = st.number_input("Lap Time Delta", min_value=-20.0, max_value=20.0, value=4.5)
    cumulative_degradation = st.number_input(
        "Cumulative Degradation",
        min_value=-100.0,
        max_value=100.0,
        value=22.0
    )

with col3:
    race_progress = st.number_input("Race Progress", min_value=0.0, max_value=1.0, value=0.65)
    normalized_tyre_life = st.number_input("Normalized Tyre Life", min_value=0.0, max_value=1.0, value=0.75)
    position_change = st.number_input("Position Change", min_value=-20, max_value=20, value=3)

compound = st.selectbox(
    "Tyre Compound",
    ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
)

input_df = pd.DataFrame({
    "Stint": [stint],
    "TyreLife": [tyre_life],
    "Position": [position],
    "LapTime (s)": [lap_time],
    "LapTime_Delta": [lap_time_delta],
    "Cumulative_Degradation": [cumulative_degradation],
    "RaceProgress": [race_progress],
    "Normalized_TyreLife": [normalized_tyre_life],
    "Position_Change": [position_change],
    "Compound": [compound]
})

st.subheader("Input Data")
st.dataframe(input_df, use_container_width=True)

st.divider()

st.header("Cloud Tests")

if st.button("Test 1: App only"):
    st.success("The Streamlit app itself is working on Cloud.")

if st.button("Test 2: Check model files"):
    try:
        st.write("Checking files...")

        if os.path.exists(MODEL_PATH):
            size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            st.success(f"Model file found: {size_mb:.2f} MB")
        else:
            st.error("Model file NOT found.")

        if os.path.exists(COLUMNS_PATH):
            size_mb = os.path.getsize(COLUMNS_PATH) / (1024 * 1024)
            st.success(f"Columns file found: {size_mb:.4f} MB")
        else:
            st.error("Columns file NOT found.")

    except Exception as e:
        st.error("File check failed.")
        st.exception(e)

if st.button("Test 3: Load columns only"):
    try:
        st.write("Before loading columns...")
        model_columns = joblib.load(COLUMNS_PATH)
        st.write("After loading columns.")
        st.success(f"Columns loaded successfully. Total columns: {len(model_columns)}")
        st.write(model_columns)
    except Exception as e:
        st.error("Loading columns failed.")
        st.exception(e)

if st.button("Test 4: Load model only"):
    try:
        st.write("Before loading model...")
        rf_model = joblib.load(MODEL_PATH)
        st.write("After loading model.")

        if hasattr(rf_model, "n_jobs"):
            rf_model.n_jobs = 1

        st.success("Model loaded successfully.")
        st.write("Model type:", type(rf_model))

        if hasattr(rf_model, "n_estimators"):
            st.write("Number of trees:", rf_model.n_estimators)

    except Exception as e:
        st.error("Loading model failed.")
        st.exception(e)

if st.button("Test 5: Full prediction"):
    try:
        st.write("Step 1: Loading columns...")
        model_columns = joblib.load(COLUMNS_PATH)
        st.write("Step 1 done.")

        st.write("Step 2: Loading model...")
        rf_model = joblib.load(MODEL_PATH)
        st.write("Step 2 done.")

        if hasattr(rf_model, "n_jobs"):
            rf_model.n_jobs = 1

        st.write("Step 3: Preparing input...")
        prepared_input = prepare_input(input_df, model_columns)
        st.write("Step 3 done.")
        st.write("Prepared input shape:", prepared_input.shape)

        st.write("Step 4: Predicting...")
        prediction = rf_model.predict(prepared_input)[0]
        st.write("Step 4 done.")

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Prediction: PIT on next lap.")
        else:
            st.success("Prediction: NO PIT on next lap.")

        if hasattr(rf_model, "predict_proba"):
            probability = rf_model.predict_proba(prepared_input)[0][1]
            st.metric("Pit Stop Probability", f"{probability * 100:.2f}%")

    except Exception as e:
        st.error("Full prediction failed.")
        st.exception(e)
