from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware

# ✅ 1. Crear app PRIMERO
app = FastAPI()

# ✅ 2. Añadir CORS (para frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción se limita
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 3. Rutas a archivos
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# ✅ 4. Cargar modelo y scaler
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    mean, std = pickle.load(f)

# ✅ 5. Esquema de entrada (JSON)
class Vivienda(BaseModel):
    metros: float
    habitaciones: float
    edad: float

# ✅ 6. Endpoint base
@app.get("/")
def home():
    return {"mensaje": "API funcionando 🚀"}

# ✅ 7. Endpoint de predicción
@app.post("/predict")
def predict(data: Vivienda):
    
    X = np.array([[data.metros, data.habitaciones, data.edad]])
    X = (X - mean) / std
    
    pred = model.predict(X)[0][0]
    
    return {
        "precio_estimado": round(pred, 2)
    }