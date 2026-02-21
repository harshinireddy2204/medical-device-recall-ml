"""
Export model.device_rpss + device names to CSV for Streamlit Cloud deployment.

Joins with vw_FDA_Device_Integrated to include device_name (k_devicename, pc_devicename,
GENERICNAME, TRADENAME) for ML Top 20 display.

Run from project root:
    python Scripts/export_device_rpss_to_csv.py

Requirements:
    pip install sqlalchemy pyodbc pandas
"""
import os
import sys

try:
    from sqlalchemy import create_engine, text
    import pandas as pd
except ImportError:
    print("Please install: pip install sqlalchemy pyodbc pandas")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "visualization", "device_rpss_sample.csv")

# Columns required by app_.py
REQUIRED_COLUMNS = [
    "PMA_PMN_NUM",
    "rpss",
    "rpss_category",
    "recall_count",
    "total_adverse_events",
    "unique-manufacturers",
    "device_class",
    "root_cause_description",
    "last_scored",
    "device_name",  # From vw_FDA_Device_Integrated for ML Top 20
]


def main():
    engine = create_engine(
        "mssql+pyodbc:///?odbc_connect="
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=FDADatabase;"
        "Trusted_Connection=yes;",
        pool_pre_ping=True,
    )

    # Join device_rpss with vw_FDA_Device_Integrated for device names
    query = """
    WITH device_names AS (
        SELECT
            PMA_PMN_NUM,
            MAX(COALESCE(NULLIF(RTRIM(k_devicename), ''), NULLIF(RTRIM(pc_devicename), ''),
                         NULLIF(RTRIM(GENERICNAME), ''), NULLIF(RTRIM(TRADENAME), ''),
                         '—')) AS device_name
        FROM dbo.vw_FDA_Device_Integrated
        WHERE PMA_PMN_NUM IS NOT NULL
        GROUP BY PMA_PMN_NUM
    )
    SELECT TOP 50000
        dr.PMA_PMN_NUM,
        dr.rpss,
        dr.rpss_category,
        dr.recall_count,
        dr.total_adverse_events,
        dr.unique_manufacturers,
        dr.device_class,
        dr.root_cause_description,
        dr.last_scored,
        COALESCE(dn.device_name, '—') AS device_name
    FROM model.device_rpss dr
    LEFT JOIN device_names dn ON dr.PMA_PMN_NUM = dn.PMA_PMN_NUM
    ORDER BY dr.rpss DESC
    """
    try:
        df = pd.read_sql(text(query), engine)
    except Exception as e:
        # Fallback: try without vw_FDA_Device_Integrated (older DB or missing view)
        print(f"Integrated view join failed: {e}. Exporting without device_name...")
        query_fallback = """
        SELECT TOP 50000
            PMA_PMN_NUM, rpss, rpss_category, recall_count, total_adverse_events,
            unique_manufacturers, device_class, root_cause_description, last_scored
        FROM model.device_rpss
        ORDER BY rpss DESC
        """
        df = pd.read_sql(text(query_fallback), engine)
        df["device_name"] = "—"

    # Rename for CSV (app_.py expects unique-manufacturers)
    if "unique_manufacturers" in df.columns:
        df = df.rename(columns={"unique_manufacturers": "unique-manufacturers"})

    # Ensure device_name exists
    if "device_name" not in df.columns:
        df["device_name"] = "—"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # Data check
    has_names = (df["device_name"].fillna("—").str.strip() != "—").sum()
    print(f"Exported {len(df):,} rows to {OUTPUT_PATH}")
    print(f"Device names available: {has_names:,} / {len(df):,}")
    print("Ready for Streamlit Cloud. See docs/DEPLOY_STEPS.md for push instructions.")


if __name__ == "__main__":
    main()
