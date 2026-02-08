# load_to_sql.py
import pandas as pd
import logging
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [INFO] %(message)s")

server = "localhost"
database = "FDADatabase"
driver = "ODBC Driver 17 for SQL Server"
table_name = "FDA_Model_Data"

conn_str = f"mssql+pyodbc://@{server}/{database}?driver={driver.replace(' ', '+')}&trusted_connection=yes"
engine = create_engine(conn_str, fast_executemany=True)

logging.info(f"Loading processed CSV into table {table_name}...")

df = pd.read_csv(r"C:\Users\harsh\FDA_pipeline\data\processed\fda_tableau_ready.csv")

df.to_sql(table_name, con=engine, if_exists="replace", index=False, low_memory =False)
logging.info(f"Successfully loaded {len(df):,} rows into {table_name}")

engine.dispose()
