import streamlit as st
import joblib
import numpy as np

# Load model and encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("RF Spectrum Classifier")

# Input fields
freq = st.number_input("Frequency (MHz)", min_value=100, max_value=30000, value=1500)
bw = st.number_input("Bandwidth (MHz)", min_value=1, max_value=1000, value=20)
power = st.number_input("Power (dBm)", min_value=-100, max_value=100, value=0)
modulation = st.selectbox("Modulation", ['BPSK', 'QPSK', 'QAM16', 'QAM64'])

# Predict button
if st.button("Classify Spectrum Band"):
    mod_encoded = label_encoder.transform([modulation])[0]
    input_data = np.array([[freq, bw, power, mod_encoded]])
    prediction = model.predict(input_data)[0]
    st.success(f"The predicted RF Spectrum Band is: **{prediction}**")
