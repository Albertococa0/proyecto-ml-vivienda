def preprocess(df):
    # features
    X = df[["metros", "habitaciones", "edad"]].values
    y = df["precio"].values.reshape(-1, 1)
    
    # normalización
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    
    X = (X - mean) / std
    
    return X, y, mean, std