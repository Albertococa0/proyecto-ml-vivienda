import numpy as np

class LinearRegressionGD:
    
    def __init__(self, lr=0.01, epochs=200):
        self.lr = lr
        self.epochs = epochs
    
    def fit(self, X, y):
        n_features = X.shape[1]
        
        self.w = np.zeros((n_features, 1))
        self.b = 0
        
        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            
            error = y_pred - y
            
            dw = (2/len(X)) * X.T @ error
            db = (2/len(X)) * np.sum(error)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
    
    def predict(self, X):
        return X @ self.w + self.b