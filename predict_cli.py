import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

# cargar modelo y scaler
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    mean, std = pickle.load(f)

# input usuario
metros = float(input("Metros: "))
habitaciones = float(input("Habitaciones: "))
edad = float(input("Edad: "))

# preparar datos
X_new = np.array([[metros, habitaciones, edad]])
X_new = (X_new - mean) / std

# predicción
precio = model.predict(X_new)

print(f"\n💰 Precio estimado: {precio[0][0]:,.2f} €")