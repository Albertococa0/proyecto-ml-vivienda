import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("data.db")

n = 200

df = pd.DataFrame({
    "metros": np.random.randint(50, 150, n),
    "habitaciones": np.random.randint(1, 5, n),
    "edad": np.random.randint(0, 30, n),
})

df["precio"] = (
    df["metros"] * 3000 +
    df["habitaciones"] * 10000 -
    df["edad"] * 500 +
    np.random.randn(n) * 10000
)

df.to_sql("viviendas", conn, if_exists="replace", index=False)

conn.close()