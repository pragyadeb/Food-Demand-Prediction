# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_log_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================
# 2. LOAD DATA
# =========================
train = pd.read_csv("train.csv")

# Sort data
train = train.sort_values(by=["center_id", "meal_id", "week"])

# =========================
# 3. FEATURE ENGINEERING
# =========================

# Price difference
train["price_diff"] = train["base_price"] - train["checkout_price"]

# Select features
features = ["num_orders", "checkout_price", "price_diff"]
data = train[features]

# =========================
# 4. NORMALIZATION
# =========================
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# =========================
# 5. CREATE SEQUENCES
# =========================
def create_sequences(data, time_steps=10):
    X, y = [], []
    
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps][0])  # num_orders
    
    return np.array(X), np.array(y)

TIME_STEPS = 10
X, y = create_sequences(data_scaled, TIME_STEPS)

# =========================
# 6. TRAIN-TEST SPLIT
# =========================
split = int(0.8 * len(X))

X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# =========================
# 7. BUILD LSTM MODEL
# =========================
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, X.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)

# =========================
# 8. TRAIN MODEL
# =========================
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# =========================
# 9. PREDICTIONS
# =========================
preds = model.predict(X_val)

# Inverse scaling
preds_full = np.zeros((len(preds), data.shape[1]))
preds_full[:, 0] = preds[:, 0]
preds_inverse = scaler.inverse_transform(preds_full)[:, 0]

y_full = np.zeros((len(y_val), data.shape[1]))
y_full[:, 0] = y_val
y_inverse = scaler.inverse_transform(y_full)[:, 0]

# =========================
# =========================
# 10. EVALUATION + GRAPH (FINAL FIX)
# =========================

import matplotlib.pyplot as plt

# Permanently fix invalid values
y_inverse = np.clip(y_inverse, 0, None)
preds_inverse = np.clip(preds_inverse, 0, None)

# Safe RMSLE using log1p
rmsle = np.sqrt(np.mean((np.log1p(preds_inverse) - np.log1p(y_inverse))**2))
print("RMSLE:", rmsle)

# Flatten for plotting
y_inverse = y_inverse.flatten()
preds_inverse = preds_inverse.flatten()

# Plot graph
plt.figure(figsize=(10,5))
plt.plot(y_inverse[:100], label="Actual")
plt.plot(preds_inverse[:100], label="Predicted")

plt.title("Actual vs Predicted Food Demand")
plt.xlabel("Time")
plt.ylabel("Number of Orders")
plt.legend()

plt.savefig("prediction_graph.png")
plt.show()

print("✅ LSTM Model Completed!")