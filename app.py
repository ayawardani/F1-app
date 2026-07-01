import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="F1 Pit Stop Prediction",
    page_icon="🏎️",
    layout="wide"
)

# -----------------------------
# Load model and columns
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "model/pitstop_model.pkl"
    columns_path = "model/model_columns.pkl"

    rf_model = joblib.load(model_path)

    # Safer for Streamlit Cloud
    if hasattr(rf_model, "n_jobs"):
        rf_model.n_jobs = 1

    model_columns = joblib.load(columns_path)

    return rf_model, model_columns


# -----------------------------
# Helper function for preprocessing
# -----------------------------
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


# -----------------------------
# App title
# -----------------------------
st.title("🏎️ Formula 1 Pit Stop Prediction App")

st.markdown("""
This app predicts whether a Formula 1 driver is likely to **pit on the next lap**
based on race strategy variables such as tyre life, position, lap time, degradation,
race progress, and tyre compound.
""")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "Single Prediction",
    "Batch CSV Prediction",
    "About the Model"
])

# ======================================================
# TAB 1: Single prediction
# ======================================================
with tab1:
    st.header("Single Race Situation Prediction")
    st.write("Enter the current race situation below.")

    col1, col2, col3 = st.columns(3)

    with col1:
        stint = st.number_input("Stint", min_value=1, max_value=10, value=1)
        tyre_life = st.number_input("Tyre Life", min_value=0, max_value=80, value=12)
        position = st.number_input("Position", min_value=1, max_value=20, value=8)

    with col2:
        lap_time = st.number_input("Lap Time (s)", min_value=40.0, max_value=200.0, value=90.0)
        lap_time_delta = st.number_input("Lap Time Delta", min_value=-20.0, max_value=20.0, value=0.5)
        cumulative_degradation = st.number_input(
            "Cumulative Degradation",
            min_value=-100.0,
            max_value=100.0,
            value=2.0
        )

    with col3:
        race_progress = st.number_input("Race Progress", min_value=0.0, max_value=1.0, value=0.45)
        normalized_tyre_life = st.number_input("Normalized Tyre Life", min_value=0.0, max_value=1.0, value=0.30)
        position_change = st.number_input("Position Change", min_value=-20, max_value=20, value=0)

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
    st.dataframe(input_df, width="stretch")

    if st.button("Predict Pit Stop"):
        try:
            with st.spinner("Generating prediction..."):
                rf_model, model_columns = load_model()
                prepared_input = prepare_input(input_df, model_columns)

                prediction = rf_model.predict(prepared_input)[0]

                # Try probability only if available
                prediction_probability = None
                if hasattr(rf_model, "predict_proba"):
                    prediction_probability = rf_model.predict_proba(prepared_input)[0][1]

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("Prediction: The driver is likely to PIT on the next lap.")
            else:
                st.success("Prediction: The driver is likely NOT to pit on the next lap.")

            if prediction_probability is not None:
                st.metric(
                    label="Pit Stop Probability",
                    value=f"{prediction_probability * 100:.2f}%"
                )
                st.progress(float(prediction_probability))

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)


# ======================================================
# TAB 2: Batch CSV prediction
# ======================================================
with tab2:
    st.header("Batch CSV Prediction")

    st.info("""
Batch CSV prediction is prepared as a future extension. 
The current stable deployed version focuses on manual single prediction.
""")

    st.markdown("""
For batch prediction, the uploaded CSV must follow the exact same input structure used by the model, including:

- Stint
- TyreLife
- Position
- LapTime (s)
- LapTime_Delta
- Cumulative_Degradation
- RaceProgress
- Normalized_TyreLife
- Position_Change
- Compound
""")


# ======================================================
# TAB 3: About model
# ======================================================
with tab3:
    st.header("About the Model")

    st.markdown("""
### Project Topic
**Predictive Race Strategy**

### Problem Type
Supervised Learning - Classification

### Prediction Target
The model predicts whether a driver will make a pit stop on the next lap.

### Target Variable
`PitNextLap`

### Model Used
Random Forest Classifier

### Main Input Features
- Stint
- TyreLife
- Position
- LapTime (s)
- LapTime_Delta
- Cumulative_Degradation
- RaceProgress
- Normalized_TyreLife
- Position_Change
- Compound

### How the App Works
The app collects race-condition inputs, applies the same preprocessing logic used during training, 
and sends the prepared input to the selected Random Forest model.

The output is a **Pit / No Pit prediction** and, when available, the **pit stop probability**.
""")
