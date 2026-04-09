import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("train.csv")

# Use relevant columns
data = data[["num_orders", "checkout_price", "base_price"]]

# Create new feature
data["price_diff"] = data["base_price"] - data["checkout_price"]

# Keep only required features
data = data[["num_orders", "checkout_price", "price_diff"]]

# =========================
# SCALING
# =========================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 🔥 SAVE SCALER (IMPORTANT)
joblib.dump(scaler, "scaler.pkl")

# =========================
# CREATE SEQUENCES
# =========================
X = []
y = []

TIME_STEPS = 1  # (keeping same as your project)

for i in range(TIME_STEPS, len(scaled_data)):
    X.append(scaled_data[i-TIME_STEPS:i])
    y.append(scaled_data[i, 0])  # num_orders

X = np.array(X)
y = np.array(y)

# =========================
# BUILD GRU MODEL
# =========================
model = Sequential()

model.add(GRU(50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# =========================
# TRAIN MODEL
# =========================
model.fit(X, y, epochs=5, batch_size=32)

# =========================
# SAVE MODEL
# =========================
model.save("gru_model.h5")

print("✅ GRU Model Completed!")
print("✅ scaler.pkl saved!")