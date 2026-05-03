import numpy as np

def evaluate(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    
    mse = np.mean((y_test - y_pred)**2)
    rmse = np.sqrt(mse)
    
    # R2
    ss_res = np.sum((y_test - y_pred)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    
    r2 = 1 - (ss_res / ss_tot)
    
    return rmse, r2