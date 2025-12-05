import numpy as np
from tensorflow.keras.models import load_model

def load_prediction_model(model_path):
    # compile=False avoids loading old optimizer state
    return load_model(model_path, compile=False)

def make_prediction(model, input_data):
    """
    Makes prediction without any scaler.
    input_data must be shaped (window_size, n_features)
    """
    # Ensure correct shape: (1, time_steps, features)
    if input_data.ndim == 2:
        input_data = np.expand_dims(input_data, axis=0)

    # Model prediction
    prediction = model.predict(input_data)

    # Return scalar
    return float(prediction[0][0])
