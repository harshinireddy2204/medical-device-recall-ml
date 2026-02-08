# ==============================================
# extract_fda_integrated_data.py
# ==============================================
import pandas as pd
import logging
from sqlalchemy import create_engine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INFO] %(message)s"
)

# Database configuration
server = "localhost"
database = "FDADatabase"
driver = "ODBC Driver 17 for SQL Server"

# SQLAlchemy connection string
conn_str = (
    f"mssql+pyodbc://@{server}/{database}"
    f"?driver={driver.replace(' ', '+')}&trusted_connection=yes"
)

# Create the engine
engine = create_engine(conn_str, fast_executemany=True)

# SQL query
query = """
SELECT *
FROM dbo.vw_FDA_Device_Integrated
"""

# Output path
output_path = r"C:\Users\harsh\FDA_pipeline\data\processed\fda_integrated_data.csv"

logging.info(" Extracting FDA integrated data in chunks...")

# Use chunked loading
chunks = pd.read_sql(query, engine, chunksize=50000)

# Write chunks progressively to CSV
with open(output_path, "w", encoding="utf-8", newline="") as f:
    first = True
    total_rows = 0
    for chunk in chunks:
        chunk.to_csv(f, index=False, mode="a", header=first)
        total_rows += len(chunk)
        first = False
        logging.info(f" Written {total_rows:,} rows so far...")

logging.info(f"Export completed successfully. File saved at: {output_path}")
engine.dispose()
