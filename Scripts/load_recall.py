import os
import json
import ijson
import pyodbc
import logging

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
JSON_FILE = r"C:\Users\harsh\FDA_pipeline\data\raw\recall\recall.json"
TABLE_NAME = "dbo.RECALL"
BATCH_SIZE = 5000

CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=FDADatabase;"
    "Trusted_Connection=yes;"
)

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def get_sql_columns(conn, table_name):
    """Fetch column info for verification"""
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name.split('.')[-1]}';
    """)
    cols = cursor.fetchall()
    cursor.close()
    return {row[0]: (row[1], row[2]) for row in cols}


def verify_nvarchar_max(conn, table_name):
    """Ensure long text columns use NVARCHAR(MAX)."""
    logging.info("Verifying NVARCHAR(MAX) columns in table...")
    cols = get_sql_columns(conn, table_name)
    bad_cols = [c for c, (dtype, length) in cols.items() if dtype == "nvarchar" and (length is not None and length < 4000)]
    if bad_cols:
        logging.warning(f"⚠️ Columns not NVARCHAR(MAX): {bad_cols}")
    else:
        logging.info("✅ All NVARCHAR columns are NVARCHAR(MAX) or DATE.")


def clean_value(value):
    """Normalize Python values for SQL insert."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value in ("", " ", None):
        return None
    return str(value).strip()


def truncate_table(conn, table_name):
    """Truncate table to avoid duplicates."""
    cursor = conn.cursor()
    logging.info(f"Truncating table {table_name}...")
    cursor.execute(f"TRUNCATE TABLE {table_name};")
    conn.commit()
    cursor.close()
    logging.info(f"Table {table_name} truncated.")


def insert_batch(conn, table_name, columns, batch):
    """Bulk insert batch with fast_executemany"""
    if not batch:
        return

    cursor = conn.cursor()
    cursor.fast_executemany = True

    col_list = ", ".join(f"[{c}]" for c in columns)
    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT INTO {table_name} ({col_list}) VALUES ({placeholders})"

    data = [tuple(row.get(c) for c in columns) for row in batch]

    try:
        cursor.executemany(sql, data)
        conn.commit()
    except Exception as e:
        conn.rollback()
        logging.error(f"Insert failed: {e}")
    finally:
        cursor.close()


# -------------------------------------------------------------------
# Main Loader
# -------------------------------------------------------------------
def load_json_stream_to_sql(conn, table_name, json_path):
    logging.info(f"Loading {json_path} into {table_name}...")

    # Verify NVARCHAR(MAX) setup
    verify_nvarchar_max(conn, table_name)

    # Parse JSON with ijson (streaming)
    with open(json_path, "r", encoding="utf-8") as f:
        parser = ijson.items(f, "results.item")

        batch = []
        total_rows = 0
        columns = None

        for row in parser:
            if not columns:
                columns = list(row.keys())
                logging.info(f"Detected columns: {columns}")

            cleaned = {k: clean_value(v) for k, v in row.items()}
            batch.append(cleaned)

            if len(batch) >= BATCH_SIZE:
                insert_batch(conn, table_name, columns, batch)
                total_rows += len(batch)
                batch = []
                logging.info(f"Inserted {total_rows} rows so far...")

        # Insert remaining
        if batch:
            insert_batch(conn, table_name, columns, batch)
            total_rows += len(batch)

        logging.info(f"✅ Finished loading {table_name}. Total rows inserted: {total_rows}")


# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
if __name__ == "__main__":
    conn = pyodbc.connect(CONN_STR)
    try:
        truncate_table(conn, TABLE_NAME)
        load_json_stream_to_sql(conn, TABLE_NAME, JSON_FILE)
    finally:
        conn.close()
