import streamlit as st
import joblib
import numpy as np
import os
import gdown

# Ganti dengan ID file Google Drive dari model yang baru
MODEL_URL = "https://drive.google.com/uc?id=GANTI_DENGAN_ID_BARU"
MODEL_PATH = "bike_model.pkl"

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = joblib.load(MODEL_PATH)

# UI aplikasi
st.title("Prediksi Jumlah Penyewaan Sepeda")

st.markdown("Masukkan data untuk melakukan prediksi:")

hr = st.slider("Jam (0-23)", 0, 23, 12)
temp = st.slider("Suhu (0.0 - 1.0)", 0.0, 1.0, 0.5)
hum = st.slider("Kelembapan (0.0 - 1.0)", 0.0, 1.0, 0.5)

if st.button("Prediksi"):
    input_data = np.array([[hr, temp, hum]])
    prediction = model.predict(input_data)
    st.success(f"Prediksi jumlah sepeda: {int(prediction[0])}")
