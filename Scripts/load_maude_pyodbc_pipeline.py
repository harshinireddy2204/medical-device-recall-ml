import os
import csv
import pandas as pd
import pyodbc
import logging
from datetime import datetime

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
CHUNKSIZE = 5000
ENCODINGS = ["latin1", "ISO-8859-1"]  # no utf-8 for MAUDE files

DATE_COLUMNS = {
    "MAUDE": [
        "DATE_RECEIVED", "DATE_REPORT", "DATE_OF_EVENT",
        "DATE_FACILITY_AWARE", "REPORT_DATE", "DATE_REPORT_TO_FDA",
        "DATE_REPORT_TO_MANUFACTURER", "DATE_MANUFACTURER_RECEIVED",
        "DEVICE_DATE_OF_MANUFACTURE", "DATE_ADDED", "DATE_CHANGED",
        "SUPPL_DATES_FDA_RECEIVED", "SUPPL_DATES_MFR_RECEIVED"
    ],
    "MAUDE_MM": [
        "DATE_RECEIVED", "DATE_REPORT", "DATE_OF_EVENT",
        "DATE_FACILITY_AWARE", "REPORT_DATE", "DATE_REPORT_TO_FDA",
        "DATE_REPORT_TO_MANUFACTURER", "DATE_MANUFACTURER_RECEIVED",
        "DEVICE_DATE_OF_MANUFACTURE", "DATE_ADDED", "DATE_CHANGED",
        "SUPPL_DATES_FDA_RECEIVED", "SUPPL_DATES_MFR_RECEIVED"
    ],
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def clean_text(val):
    """Strip, normalize whitespace, return None if empty"""
    if pd.isna(val):
        return None
    val = str(val).strip()
    return val if val else None

def clean_date(val):
    """Convert strings to datetime.date, return None if invalid"""
    if pd.isna(val) or not str(val).strip():
        return None
    try:
        return pd.to_datetime(str(val).strip(), errors="coerce").date()
    except Exception:
        return None

def get_column_maxlen(conn, table_name):
    """Fetch max length for each NVARCHAR column in a table"""
    query = """
    SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = ?
    """
    cursor = conn.cursor()
    cursor.execute(query, (table_name,))
    schema_info = {}
    for row in cursor.fetchall():
        col, dtype, maxlen = row
        if dtype in ("nvarchar", "varchar"):
            schema_info[col] = maxlen  # -1 means NVARCHAR(MAX)
    cursor.close()
    return schema_info

def truncate_value(val, col, maxlen):
    """Truncate value if it exceeds SQL column length"""
    if val is None or maxlen is None or maxlen == -1:
        return val
    val = str(val)
    return val[:maxlen] if len(val) > maxlen else val

# ---------------------------------------------------------------------
# Bulk Insert with pyodbc
# ---------------------------------------------------------------------
def bulk_insert_with_pyodbc(conn, table_name, df, col_maxlens):
    cursor = conn.cursor()
    cursor.fast_executemany = True

    columns = ', '.join([f'[{c}]' for c in df.columns])
    placeholders = ', '.join(['?' for _ in df.columns])
    sql = f"INSERT INTO dbo.[{table_name}] ({columns}) VALUES ({placeholders})"

    # Truncate where needed
    for col in df.columns:
        if col in col_maxlens:
            maxlen = col_maxlens[col]
            df[col] = df[col].map(lambda v: truncate_value(v, col, maxlen))

    data = [tuple(row) for row in df.itertuples(index=False, name=None)]

    try:
        cursor.executemany(sql, data)
        conn.commit()
        return len(data)
    except Exception as e:
        conn.rollback()
        logging.error(f"Insert failed: {e}")
        return 0
    finally:
        cursor.close()

# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------
def load_file(conn, table_name, filepath):
    logging.info(f"Starting load for {filepath} -> {table_name}")

    # Detect encoding
    encoding_to_use = None
    for enc in ENCODINGS:
        try:
            pd.read_csv(filepath, sep="|", dtype=str, encoding=enc,
                        nrows=5, quoting=csv.QUOTE_NONE, engine="python")
            encoding_to_use = enc
            logging.info(f"Encoding detected: {enc}")
            break
        except UnicodeDecodeError:
            continue

    if not encoding_to_use:
        logging.error(f"Failed to detect encoding for {filepath}")
        return

    # Get SQL schema column lengths
    col_maxlens = get_column_maxlen(conn, table_name)

    total_rows = 0
    for chunk in pd.read_csv(filepath, sep="|", dtype=str,
                             encoding=encoding_to_use,
                             chunksize=CHUNKSIZE,
                             quoting=csv.QUOTE_NONE,
                             engine="python"):
        # Clean text
        for col in chunk.columns:
            chunk[col] = chunk[col].map(clean_text)

        # Clean dates
        for col in DATE_COLUMNS.get(table_name, []):
            if col in chunk.columns:
                chunk[col] = chunk[col].map(clean_date)

        inserted = bulk_insert_with_pyodbc(conn, table_name, chunk, col_maxlens)
        total_rows += inserted
        logging.info(f"Inserted {inserted} rows (total {total_rows})")

    logging.info(f"Finished load for {table_name}. Total rows: {total_rows}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    conn_str = os.getenv("FDA_DB_CONN")
    if not conn_str:
        conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=localhost;"
            "DATABASE=FDADatabase;"
            "Trusted_Connection=yes;"
        )

    conn = pyodbc.connect(conn_str)

    try:
        base = r"C:\Users\harsh\FDA_pipeline\data\raw\maude"
        files = {
            "MAUDE": os.path.join(base, "maude_tilldate2025.txt"),
            "MAUDE_MM": os.path.join(base, "maude_mm.txt"),
        }
        for tbl, path in files.items():
            load_file(conn, tbl, path)
    finally:
        conn.close()
