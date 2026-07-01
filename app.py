import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="F1 Pit Stop Prediction",
    page_icon="🏎️",
    layout="wide"
)

# -----------------------------
# Train model from CSV
# -----------------------------
@st.cache_resource
def train_model():
    data_path = "f1_strategy_dataset_v4.csv"

    df = pd.read_csv(
        data_path,
        engine="python",
        on_bad_lines="skip"
    )

    df.columns = df.columns.str.strip()

    feature_columns = [
        "Stint",
        "TyreLife",
        "Position",
        "LapTime (s)",
        "LapTime_Delta",
        "Cumulative_Degradation",
        "RaceProgress",
        "Normalized_TyreLife",
        "Position_Change",
        "Compound"
    ]

    target_column = "PitNextLap"

    df = df.dropna(subset=feature_columns + [target_column])

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    X_encoded = pd.get_dummies(X, columns=["Compound"], drop_first=True)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=1
    )

    model.fit(X_encoded, y)

    model_columns = X_encoded.columns.tolist()

    return model, model_columns


# -----------------------------
# Preprocess input
# -----------------------------
def prepare_input(input_df, model_columns):
    input_df = input_df.copy()
    input_df.columns = input_df.columns.str.strip()

    input_encoded = pd.get_dummies(input_df, columns=["Compound"], drop_first=True)
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

    if st.button("Predict Pit Stop"):
        try:
            with st.spinner("Training model and generating prediction..."):
                rf_model, model_columns = train_model()
                prepared_input = prepare_input(input_df, model_columns)

                prediction = rf_model.predict(prepared_input)[0]
                probability = rf_model.predict_proba(prepared_input)[0][1]

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("Prediction: The driver is likely to PIT on the next lap.")
            else:
                st.success("Prediction: The driver is likely NOT to pit on the next lap.")

            st.metric(
                label="Pit Stop Probability",
                value=f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)


# ======================================================
# TAB 2: Batch CSV prediction
# ======================================================
with tab2:
    st.header("Batch CSV Prediction")

    st.info("""
The deployed version focuses on single race-situation prediction.
Batch CSV prediction can be added as a future extension.
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

### How the Cloud Version Works
For Streamlit Cloud, the app trains the Random Forest model directly from the dataset file included in the GitHub repository.

This avoids the cloud compatibility issue that occurred when loading the locally saved `.pkl` model file.
""")
