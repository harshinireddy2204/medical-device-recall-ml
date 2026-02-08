# ==============================================
# export_to_tableau_ready.py
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

# SQLAlchemy connection
conn_str = (
    f"mssql+pyodbc://@{server}/{database}"
    f"?driver={driver.replace(' ', '+')}&trusted_connection=yes"
)
engine = create_engine(conn_str, fast_executemany=True)

# Tableau-ready summarized query
query = """
SELECT 
    PMA_PMN_NUM,
    adverse_event_count,
    total_adverse_flag,
    total_product_problem_flag,
    unique_manufacturers,
    unique_distributors,
    first_event_date,
    last_event_date,
    recall_status,
    reason_for_recall,
    product_quantity,
    DEVICECLASS,
    MEDICALSPECIALTY,
    REVIEW_PANEL
FROM dbo.vw_FDA_Device_Integrated
"""

output_path = r"C:\Users\harsh\FDA_pipeline\data\processed\fda_tableau_ready.csv"

logging.info("Extracting Tableau-ready FDA data in chunks...")

# Chunked loading
chunks = pd.read_sql(query, engine, chunksize=50000)

with open(output_path, "w", encoding="utf-8", newline="") as f:
    first = True
    total_rows = 0
    for chunk in chunks:
        # Convert dates and clean
        chunk["first_event_date"] = pd.to_datetime(chunk["first_event_date"], errors="coerce")
        chunk["last_event_date"] = pd.to_datetime(chunk["last_event_date"], errors="coerce")

        chunk.to_csv(f, index=False, mode="a", header=first)
        total_rows += len(chunk)
        first = False
        logging.info(f"Written {total_rows:,} rows so far...")

logging.info(f"Tableau-ready data exported successfully: {output_path}")
engine.dispose()
