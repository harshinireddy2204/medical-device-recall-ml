#!/usr/bin/env python3
"""
RPSS Pipeline — Final Stable Version
- Reads from vw_FDA_Device_Integrated in chunks
- Aggregates safely (no max() on strings)
- Uses root_cause_description
- Computes RPSS
- Writes to staging then MERGE into final table
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import logging, sys

# ===========================================================
# SQL CONFIG
# ===========================================================
SQL_CONN_STR = (
    "mssql+pyodbc:///?odbc_connect="
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=FDADatabase;"
    "Trusted_Connection=yes;"
)
engine = create_engine(SQL_CONN_STR)

CFG = {
    "staging": "model.device_rpss_staging",
    "target": "model.device_rpss"
}

# ===========================================================
# LOGGING
# ===========================================================
logger = logging.getLogger("rpss")
logger.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(h)

# ===========================================================
# ROOT CAUSE SEVERITY
# ===========================================================
ROOT_CAUSE_MAP = {
    "Software design": 0.95,
    "Software": 0.95,
    "Device Design": 0.90,
    "Design": 0.90,
    "Process control": 0.80,
    "Manufacturing": 0.80,
    "Nonconforming Material/Component": 0.75,
    "Packaging": 0.60,
    "Labeling": 0.45,
    "Mixed-up of materials/components": 0.40,
    "Equipment maintenance": 0.40,
    "Other": 0.50,
    "Under Investigation by firm": 0.55
}

def map_root_cause(val):
    if not isinstance(val, str):
        return ROOT_CAUSE_MAP["Other"]
    for k, score in ROOT_CAUSE_MAP.items():
        if k.lower() in val.lower():
            return score
    return ROOT_CAUSE_MAP["Other"]

# ===========================================================
# NORMALIZATION
# ===========================================================
def normalize_series(s):
    if s.empty:
        return s
    mn, mx = s.min(), s.max()
    if mn == mx:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn + 1e-9)

# ===========================================================
# RPSS SCORING
# ===========================================================
def compute_rpss(df):

    df["recurrence_raw"] = df["recall_count"]
    df["recurrence_score"] = normalize_series(df["recurrence_raw"])

    df["root_cause_raw"] = df["root_cause_description"].apply(map_root_cause)
    df["reason_severity_score"] = normalize_series(df["root_cause_raw"])

    df["density_raw"] = df["total_adverse_events"] / df["unique_manufacturers"].clip(lower=1)
    df["adverse_density_score"] = normalize_series(df["density_raw"])

    df["first_event_date"] = pd.to_datetime(df["first_event_date"], errors="coerce")
    df["last_event_date"] = pd.to_datetime(df["last_event_date"], errors="coerce")

    df["timespan_days"] = (df["last_event_date"] - df["first_event_date"]).dt.days.fillna(0)
    df["recency_raw"] = df["timespan_days"].apply(lambda x: 0 if x <= 0 else 1 / x)
    df["recency_score"] = normalize_series(df["recency_raw"])

    def class_score(v):
        try:
            v = int(v)
            return {3: 1.0, 2: 0.7, 1: 0.4}.get(v, 0.5)
        except:
            return 0.5

    df["device_class_raw"] = df["device_class"].apply(class_score)
    df["device_class_score"] = normalize_series(df["device_class_raw"])

    df["rpss"] = (
        df["recurrence_score"] * 0.30 +
        df["reason_severity_score"] * 0.30 +
        df["adverse_density_score"] * 0.20 +
        df["recency_score"] * 0.10 +
        df["device_class_score"] * 0.10
    ).clip(0, 1)

    def cat(x):
        if x <= 0.25: return "Low"
        if x <= 0.50: return "Medium"
        if x <= 0.75: return "High"
        return "Critical"

    df["rpss_category"] = df["rpss"].apply(cat)

    return df[[
        "PMA_PMN_NUM",
        "recall_count",
        "total_adverse_events",
        "unique_manufacturers",
        "device_class",
        "root_cause_description",
        "rpss",
        "rpss_category"
    ]]

# ===========================================================
# MAIN PIPELINE
# ===========================================================
def main(chunk_size=100000):

    log = logger
    log.info("=== RPSS pipeline started ===")

    staging_schema, staging_tbl = CFG["staging"].split(".")

    with engine.begin() as conn:
        conn.execute(text(f"IF OBJECT_ID('{CFG['staging']}', 'U') IS NOT NULL DROP TABLE {CFG['staging']}"))
        log.info("Staging table cleared.")

    query = "SELECT * FROM dbo.vw_FDA_Device_Integrated;"

    for idx, chunk in enumerate(pd.read_sql(query, engine, chunksize=chunk_size), start=1):
        log.info(f"Chunk {idx}: {len(chunk)} rows")

        chunk = chunk[chunk["PMA_PMN_NUM"].notna()]

        for col in [
            "adverse_event_count", "unique_manufacturers",
            "unique_distributors", "DEVICECLASS"
        ]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        agg = chunk.groupby("PMA_PMN_NUM").agg(
            recall_count=("cfres_id", "nunique"),
            total_adverse_events=("adverse_event_count", "sum"),
            unique_manufacturers=("unique_manufacturers", "sum"),
            device_class=("DEVICECLASS", "max"),
            first_event_date=("first_event_date", "min"),
            last_event_date=("last_event_date", "max"),
            root_cause_description=("root_cause_description",
                lambda s: s.dropna().iloc[0] if not s.dropna().empty else "Other")
        ).reset_index()

        rpss = compute_rpss(agg)

        rpss.to_sql(
            staging_tbl,
            schema=staging_schema,
            con=engine,
            if_exists="append",
            index=False
        )

    log.info("Merging into final table...")

    merge_sql = """
WITH src_dedup AS (
    SELECT
        PMA_PMN_NUM,
        rpss,
        rpss_category,
        recall_count,
        total_adverse_events,
        unique_manufacturers,
        device_class,
        root_cause_description,
        ROW_NUMBER() OVER (
            PARTITION BY PMA_PMN_NUM 
            ORDER BY PMA_PMN_NUM
        ) AS rn
    FROM model.device_rpss_staging
),
src_clean AS (
    SELECT
        PMA_PMN_NUM,
        rpss,
        rpss_category,
        recall_count,
        total_adverse_events,
        unique_manufacturers,
        device_class,
        root_cause_description
    FROM src_dedup
    WHERE rn = 1
)

MERGE model.device_rpss AS tgt
USING src_clean AS src
ON tgt.PMA_PMN_NUM = src.PMA_PMN_NUM

WHEN MATCHED THEN UPDATE SET
    tgt.rpss                 = src.rpss,
    tgt.rpss_category        = src.rpss_category,
    tgt.recall_count         = src.recall_count,
    tgt.total_adverse_events = src.total_adverse_events,
    tgt.unique_manufacturers = src.unique_manufacturers,
    tgt.device_class         = src.device_class,
    tgt.root_cause_description = src.root_cause_description,
    tgt.last_scored          = SYSUTCDATETIME()

WHEN NOT MATCHED THEN INSERT (
    PMA_PMN_NUM,
    rpss, rpss_category,
    recall_count,
    total_adverse_events,
    unique_manufacturers,
    device_class,
    root_cause_description,
    last_scored
)
VALUES (
    src.PMA_PMN_NUM,
    src.rpss, src.rpss_category,
    src.recall_count,
    src.total_adverse_events,
    src.unique_manufacturers,
    src.device_class,
    src.root_cause_description,
    SYSUTCDATETIME()
);
"""


    with engine.begin() as conn:
        conn.execute(text(merge_sql))

    log.info("=== RPSS pipeline finished successfully ===")


if __name__ == "__main__":
    main()
