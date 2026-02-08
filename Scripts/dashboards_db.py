import pyodbc
import pandas as pd

def get_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=FDADatabase;"
        "Trusted_Connection=yes;"
    )

def load_view(view_name):
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM {view_name}", conn)
    conn.close()
    return df
