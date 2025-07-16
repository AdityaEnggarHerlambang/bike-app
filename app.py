import streamlit as st
import joblib
import numpy as np
import os
import gdown

MODEL_URL = "https://drive.google.com/uc?id=1FJW_BL2jsDR2XfAVXbTrSVjLlo0fSLPx"
MODEL_PATH = "bike_model.pkl"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = joblib.load("bike_model.pkl")

st.title("Prediksi Jumlah Penyewaan Sepeda")

fitur = st.slider("Masukkan nilai fitur", 0, 100, 50)  # Sesuai dengan data training

if st.button("Prediksi"):
    input_data = np.array([[fitur]])
    prediction = model.predict(input_data)
    st.success(f"Prediksi jumlah sepeda: {int(prediction[0])}")
