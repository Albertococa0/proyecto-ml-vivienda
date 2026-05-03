import numpy as np
import pickle
import os

# ruta base del proyecto
BASE_DIR = os.path.dirname(__file__)

# rutas absolutas
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

# cargar modelo
with open(model_path, "rb") as f:
    model = pickle.load(f)

# cargar scaler
with open(scaler_path, "rb") as f:
    mean, std = pickle.load(f)

# datos nuevos
X_new = np.array([[90, 3, 10]])

# normalizar
X_new = (X_new - mean) / std

# predecir
precio = model.predict(X_new)

print("Precio predicho:", precio[0][0])