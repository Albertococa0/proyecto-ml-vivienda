from sklearn.model_selection import train_test_split
from src.model import LinearRegressionGD   # 👈 IMPORT CORREGIDO

def train_model(X, y):
    
    # dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # inicializar modelo
    model = LinearRegressionGD()
    
    # entrenar
    model.fit(X_train, y_train)
    
    return model, X_test, y_test