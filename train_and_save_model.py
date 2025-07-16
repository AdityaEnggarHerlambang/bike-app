import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Data training dengan 3 fitur
data = {
    "hr": [1, 2, 3, 4, 5, 6, 7],
    "temp": [0.2, 0.4, 0.6, 0.8, 0.5, 0.3, 0.7],
    "hum": [0.3, 0.5, 0.7, 0.4, 0.2, 0.6, 0.8],
    "target": [10, 20, 30, 40, 50, 60, 70]
}

df = pd.DataFrame(data)

X = df[["hr", "temp", "hum"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "bike_model.pkl")

print("Model berhasil disimpan ke bike_model.pkl")
