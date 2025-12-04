import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from prediction_helper import load_prediction_model, load_scaler, make_prediction

# --- Page Config ---
st.set_page_config(
    page_title="Weather Forecaster",
    page_icon="g",
    layout="wide"
)

st.title("Weather Temperature Predictor (LSTM)")
st.markdown("""
Enter the required weather attributes, and the model will predict the **next temperature value**.
""")

# --- Features to Ask the User ---
FEATURES = [
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed",
    "cloud_cover",
    "sunshine_duration"
]

# --- Sidebar: Model Config ---
st.sidebar.header("Configuration")
window_size = st.sidebar.number_input(
    "Lookback Window Size (Time Steps)",
    min_value=1,
    value=24
)

# --- Load Model & Scalers ---
try:
    model = load_prediction_model('model.keras')
    scaler_X = load_scaler('scaler_X.pkl')
    scaler_y = load_scaler('scaler_y.pkl')
    st.sidebar.success("Model and scalers loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model/scaler: {e}")
    st.stop()

# --- User Inputs Form ---
st.subheader("Enter Weather Attributes")

inputs = {}

cols = st.columns(3)
for i, feature in enumerate(FEATURES):
    with cols[i % 3]:
        inputs[feature] = st.number_input(
            f"{feature.replace('_', ' ').title()}",
            value=0.0,
            format="%.4f"
        )

st.info("These values represent the **most recent** measurements. The model needs them to forecast the next one.")

# --- Prediction ---
if st.button("Predict Temperature"):

    try:
        # Convert dict → numeric vector
        input_vector = np.array(list(inputs.values()), dtype=float)

        # Repeat the last entry to form window_size sequence
        # Example: LSTM expects (window_size, num_features)
        X_window = np.tile(input_vector, (window_size, 1))

        prediction = make_prediction(
            model, scaler_X, scaler_y, X_window, window_size
        )

        st.success("Prediction complete!")

        st.metric("Predicted Temperature", f"{prediction:.4f}")

        # Optional small plot
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.scatter([0], [prediction], color="red", s=150)
        ax.set_title("Predicted Temperature")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction error: {e}")
