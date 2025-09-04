import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import os

# ------------------------------
# Step 1: Load Dataset
# ------------------------------
csv_path = r"E:\tomato\archive\future\tomato_disease_full_5000_1decimal.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ Dataset not found at {csv_path}")

df = pd.read_csv(
    csv_path,
    header=None,
    names=["Temperature", "Humidity", "SoilMoisture", "Disease"],
    skiprows=1
)

# Convert numeric columns
df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
df["Humidity"] = pd.to_numeric(df["Humidity"], errors="coerce")
df["SoilMoisture"] = pd.to_numeric(df["SoilMoisture"], errors="coerce")
df = df.dropna()

df['Disease'] = df['Disease'].astype(str).str.strip()
print("✅ Data sample:\n", df.head())

# ------------------------------
# Step 2: Prepare Data
# ------------------------------
X = df[["Temperature", "Humidity", "SoilMoisture"]].values
y = df["Disease"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, "label_encoder_cnn.pkl")

y_categorical = to_categorical(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler_cnn.pkl")

X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_categorical, test_size=0.2, random_state=42
)

# ------------------------------
# Step 3: Build CNN Model
# ------------------------------
model = Sequential([
    Conv1D(32, kernel_size=2, activation='relu', padding="same", input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=1),  # keep dimensions safe
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------
# Step 4: Train Model
# ------------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {accuracy*100:.2f}%")

model.save("tomato_disease_cnn.h5")
print("✅ Model saved as tomato_disease_cnn.h5")

# ------------------------------
# Step 5: Prediction Functions
# ------------------------------
def predict_disease(temp, hum, soil):
    model = tf.keras.models.load_model("tomato_disease_cnn.h5")
    scaler = joblib.load("scaler_cnn.pkl")
    le = joblib.load("label_encoder_cnn.pkl")

    X_input = np.array([[temp, hum, soil]])
    X_scaled = scaler.transform(X_input)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    pred_probs = model.predict(X_reshaped)
    pred_class = np.argmax(pred_probs, axis=1)
    return le.inverse_transform(pred_class)[0]

def predict_future(forecast_data):
    model = tf.keras.models.load_model("tomato_disease_cnn.h5")
    scaler = joblib.load("scaler_cnn.pkl")
    le = joblib.load("label_encoder_cnn.pkl")

    X_scaled = scaler.transform(forecast_data)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    pred_probs = model.predict(X_reshaped)
    pred_classes = np.argmax(pred_probs, axis=1)
    return le.inverse_transform(pred_classes)

# ------------------------------
# Step 6: Example Usage
# ------------------------------
print("➡️ Example Current Prediction:", predict_disease(28.5, 70.0, 20.0))

future_forecast = [
    [29.0, 70.0, 25.0],
    [30.5, 65.0, 28.0],
    [31.0, 68.0, 30.0],
]
print("➡️ Future Predictions:", predict_future(future_forecast))
