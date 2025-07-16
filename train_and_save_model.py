
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Contoh data dummy: bisa diganti dengan dataset asli
# Misalnya fitur = jumlah pengunjung, target = jumlah sepeda terjual
data = {
    "fitur": [10, 20, 30, 40, 50, 60, 70],
    "target": [15, 30, 45, 60, 75, 90, 105]
}

df = pd.DataFrame(data)

# Pisahkan fitur dan target
X = df[["fitur"]]
y = df["target"]

# Split (opsional, tapi umum)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model
model = LinearRegression()
model.fit(X_train, y_train)

# Simpan model ke file .pkl
joblib.dump(model, "bike_model.pkl")

print("Model berhasil disimpan ke bike_model.pkl")
