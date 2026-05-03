from src.load_data import load_data
from src.preprocess import preprocess
from src.train import train_model
from src.evaluate import evaluate

import pickle
import os

def main():
    
    # 1. cargar datos
    df = load_data()
    
    # 2. preprocess
    X, y, mean, std = preprocess(df)
    
    # 3. entrenar
    model, X_test, y_test = train_model(X, y)
    
    # 4. evaluar
    rmse, r2 = evaluate(model, X_test, y_test)
    
    print("RMSE:", rmse)
    print("R2:", r2)

    # 🔥 asegurar carpeta models
    os.makedirs("models", exist_ok=True)

    # 🔥 guardar modelo
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # 🔥 BONUS PRO: guardar normalización
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump((mean, std), f)


if __name__ == "__main__":
    main()