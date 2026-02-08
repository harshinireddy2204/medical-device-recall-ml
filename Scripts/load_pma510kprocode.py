# load_pma510kprocode_fixed_encoding.py
import os
import io
import pandas as pd
from sqlalchemy import create_engine, text

# ----------------------
# DB connection (edit if needed)
# ----------------------
SQL_CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=FDADatabase;"
    "Trusted_Connection=yes;"
)
engine = create_engine("mssql+pyodbc:///?odbc_connect=" + SQL_CONN_STR.replace(" ", "%20"))

# ----------------------
# Files to load (update paths if needed)
# ----------------------
FILES = {
    "dbo.PMA": r"C:\Users\harsh\FDA_pipeline\data\raw\pma\pma.txt",
    "dbo.Premarket510k": r"C:\Users\harsh\FDA_pipeline\data\raw\510k\premarket_510k.txt",
    "dbo.ProductCode": r"C:\Users\harsh\FDA_pipeline\data\raw\prodclass\productcode class.txt",
}

# ----------------------
# Helper: try multiple encodings, fallback to decode-with-replace
# ----------------------
def try_read_csv(filepath, sep="|", encodings=None, chunksize=None, **pd_kwargs):
    if encodings is None:
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1", "utf-16"]

    last_exc = None
    for enc in encodings:
        try:
            print(f"Trying encoding: {enc}")
            if chunksize:
                # return iterator if using chunksize
                return pd.read_csv(filepath, sep=sep, encoding=enc, chunksize=chunksize, low_memory=False, dtype=str, **pd_kwargs)
            else:
                return pd.read_csv(filepath, sep=sep, encoding=enc, low_memory=False, dtype=str, **pd_kwargs)
        except UnicodeDecodeError as e:
            print(f"  → {enc} failed with UnicodeDecodeError: {e}")
            last_exc = e
        except Exception as e:
            # parser errors or other issues may appear — show and try next encoding
            print(f"  → {enc} failed with: {type(e).__name__}: {e}")
            last_exc = e

    # Final fallback: read bytes and decode with replacement of invalid chars
    print("All encodings failed — falling back to binary read + decode(errors='replace').")
    with open(filepath, "rb") as f:
        raw = f.read()
    # attempt utf-8 replace (safe, preserves the rest of the content)
    decoded = raw.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(decoded), sep=sep, low_memory=False, dtype=str, **pd_kwargs)


# ----------------------
# Loader: delete and insert (supports large files via chunksize)
# ----------------------
def load_simple(table, filepath, sep="|", chunksize=None):
    schema, table_name = table.split(".") if "." in table else ("dbo", table)
    print(f"\nLoading {filepath} into {table} ...")

    # Delete old records (SQLAlchemy 2.0-compatible)
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table}"))

    try:
        reader = try_read_csv(filepath, sep=sep, chunksize=chunksize)
    except Exception as e:
        print(f"Failed to open file with all strategies: {type(e).__name__}: {e}")
        raise

    # If chunksize was used, reader is an iterator
    if chunksize:
        rows_total = 0
        for chunk in reader:
            chunk.to_sql(table_name, engine, schema=schema, if_exists="append", index=False)
            rows_total += len(chunk)
            print(f"  appended chunk: {len(chunk)} rows (total so far: {rows_total})")
        print(f"Inserted {rows_total} rows into {table}")
    else:
        # reader is a DataFrame
        df = reader
        df.to_sql(table_name, engine, schema=schema, if_exists="append", index=False)
        print(f"Inserted {len(df)} rows into {table}")


# ----------------------
# Main: run loaders
# ----------------------
if __name__ == "__main__":
    # For PMA/510k/ProductCode these files are usually small — no chunking by default.
    for tbl, path in FILES.items():
        if not os.path.exists(path):
            print(f"File not found: {path} — skipping {tbl}")
            continue
        load_simple(tbl, path, sep="|", chunksize=None)

    print("\nDone.")
