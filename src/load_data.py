import sqlite3
import pandas as pd

def load_data():
    conn = sqlite3.connect("data.db")
    
    query = """
    SELECT metros, habitaciones, edad, precio
    FROM viviendas
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    return df
