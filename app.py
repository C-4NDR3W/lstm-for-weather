import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from prediction_helper import load_prediction_model, make_prediction

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

# --- Features ---
FEATURES = [
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "cloud_cover",
    "sunshine_duration"
]

# --- Sidebar ---
st.sidebar.header("Configuration")
window_size = st.sidebar.number_input(
    "Lookback Window Size (Time Steps)",
    min_value=1,
    value=24
)

# --- Load Model ---
try:
    model = load_prediction_model("model.keras")
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# --- Input Form ---
st.subheader("Enter Weather Attributes")

inputs = {}
cols = st.columns(3)

for i, feature in enumerate(FEATURES):
    with cols[i % 3]:
        inputs[feature] = st.number_input(
            f"{feature.replace('_', ' ').title()}",
            value=0.0,
            format="%.2f"
        )

st.info("These values represent the **most recent** measurements.")

st.sidebar.write("Model input shape:", model.input_shape)

# --- Prediction ---
if st.button("Predict Temperature"):

    try:
        input_vector = np.array(list(inputs.values()), dtype=float)

        # Build (window_size, num_features)
        X_window = np.tile(input_vector, (window_size, 1))

        prediction = make_prediction(model, X_window)

        st.success("Prediction complete!")
        st.metric("Predicted Temperature", f"{prediction:.4f}")

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.scatter([0], [prediction], s=150)
        ax.set_title("Predicted Temperature")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction error: {e}")
