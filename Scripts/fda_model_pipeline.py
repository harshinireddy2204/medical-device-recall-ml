"""fda_model_pipeline.py

End-to-end pipeline:
  • Extract from FDADatabase.dbo.vw_FDA_Device_Integrated
  • Feature engineering, optional model training
  • Score dataset and write to SQL Server table
  • Save model artifact and feature importances
"""

import os
import json
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

CONFIG = {
    "server": "localhost",
    "database": "FDADatabase",
    "driver": "ODBC Driver 17 for SQL Server",
    "trusted_connection": True,
    "schema": "model",  # change to 'dbo' if schema access restricted
    "staging_table": "device_risk_scores_staging",
    "target_table": "device_risk_scores",
    "view_name": "dbo.vw_FDA_Device_Integrated",
    "model_path": "rf_model_v1.joblib",
    "feature_importance_table": "model_feature_importance",
    "train_model": True,
    "random_seed": 42,
    "n_estimators": 200,
    "probability_threshold": 0.5
}

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Database engine creation
# ------------------------------------------------------------------------------

def get_engine(cfg: dict):
    """Create and return SQLAlchemy engine based on configuration."""
    driver = cfg["driver"]
    driver_param = driver.replace(" ", "+")
    if cfg.get("trusted_connection", True):
        conn_str = (
            f"mssql+pyodbc://@{cfg['server']}/{cfg['database']}"
            f"?driver={driver_param}&Trusted_Connection=yes"
        )
    else:
        user = cfg["username"]
        pwd = cfg["password"]
        conn_str = (
            f"mssql+pyodbc://{user}:{pwd}@{cfg['server']}/{cfg['database']}"
            f"?driver={driver_param}"
        )
    engine = create_engine(conn_str, fast_executemany=True)
    log.info("SQLAlchemy engine created for database: %s", cfg["database"])
    return engine

# ------------------------------------------------------------------------------
# Step 1: Extraction
# ------------------------------------------------------------------------------

def extract_view(engine, view_name: str, chunksize: int = 100000) -> pd.DataFrame:
    """
    Extracts data from the specified view into a pandas DataFrame.
    Loads in chunks to handle large datasets efficiently.
    """
    log.info("Extracting data from view in chunks: %s", view_name)
    query = f"SELECT * FROM {view_name};"

    chunks = []
    total_rows = 0

    with engine.connect() as conn:
        for chunk in pd.read_sql(text(query), conn, chunksize=chunksize):
            chunk_rows = len(chunk)
            total_rows += chunk_rows
            chunks.append(chunk)
            log.info("Loaded chunk with %d rows (total so far: %d)", chunk_rows, total_rows)

    df = pd.concat(chunks, ignore_index=True)
    log.info("Finished loading all chunks. Total rows: %d", len(df))
    return df


# ------------------------------------------------------------------------------
# Step 2: Feature engineering
# ------------------------------------------------------------------------------

def feature_engineer(
    df: pd.DataFrame
) -> (pd.DataFrame, pd.DataFrame, list):
    """Perform cleaning and feature engineering, returning original DF, feature matrix, and feature columns list."""
    log.info("Starting feature engineering")
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in ("first_event_date", "last_event_date", "event_date_initiated"):
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "first_event_date" in df and "last_event_date" in df:
        df["time_on_market_years"] = (
            df["last_event_date"] - df["first_event_date"]
        ).dt.days / 365.25
        df["time_on_market_years"].fillna(0.0, inplace=True)
    else:
        df["time_on_market_years"] = 0.0

    numeric_cols = [
        "adverse_event_count",
        "total_adverse_flag",
        "total_product_problem_flag",
        "unique_manufacturers",
        "unique_distributors"
    ]
    for c in numeric_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["adverse_per_year"] = df.apply(
        lambda row: (
            row.adverse_event_count / row.time_on_market_years
            if row.time_on_market_years > 0 else row.adverse_event_count
        ), axis=1
    )

    for cat_col in ("DEVICECLASS", "REVIEW_PANEL", "MEDICALSPECIALTY"):
        if cat_col in df:
            df[cat_col] = df[cat_col].fillna("Unknown").astype(str)
        else:
            df[cat_col] = "Unknown"

    feature_cols = [
        "adverse_event_count",
        "adverse_per_year",
        "total_product_problem_flag",
        "unique_manufacturers",
        "time_on_market_years",
        "DEVICECLASS",
        "REVIEW_PANEL"
    ]

    for c in feature_cols:
        if c not in df:
            df[c] = 0

    X = df[feature_cols].copy()
    X = pd.get_dummies(X, columns=["DEVICECLASS", "REVIEW_PANEL"], dummy_na=True)
    X.fillna(0, inplace=True)

    log.info("Feature matrix shape: %s", X.shape)
    return df, X, feature_cols

# ------------------------------------------------------------------------------
# Step 3: Label creation
# ------------------------------------------------------------------------------

def create_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary label 'recall_within_5yrs' if recall date exists within five years of first_event_date."""
    if "event_date_initiated" in df and "first_event_date" in df:
        df["event_date_initiated"] = pd.to_datetime(
            df["event_date_initiated"], errors="coerce"
        )
        df["days_to_recall"] = (
            df["event_date_initiated"] - df["first_event_date"]
        ).dt.days
        df["recall_within_5yrs"] = df["days_to_recall"].apply(
            lambda x: 1 if (pd.notna(x) and 0 <= x <= (365.25 * 5)) else 0
        )
        log.info("Recall label created based on event_date_initiated.")
    else:
        df["recall_within_5yrs"] = 0
        log.warning("Recall label columns missing; defaulting to 0.")
    return df

# ------------------------------------------------------------------------------
# Step 4: Train model
# ------------------------------------------------------------------------------

def train_model(
    X: pd.DataFrame, y: pd.Series, cfg: dict
) -> Pipeline:
    """Train RandomForest model and return trained pipeline."""
    log.info("Training RandomForest classifier")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            random_state=cfg["random_seed"],
            n_jobs=-1
        ))
    ])

    if y.sum() == 0:
        log.warning("No positive cases in target variable; model may not be valid.")

    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True,
                             random_state=cfg["random_seed"])
        scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        log.info("CV ROC AUC scores: %s", np.round(scores, 4))
        log.info("Mean ROC AUC: %.4f", float(scores.mean()))
    except Exception as err:
        log.warning("Cross-validation failed: %s", err)

    pipeline.fit(X, y)
    return pipeline

# ------------------------------------------------------------------------------
# Step 5: Score & prepare output
# ------------------------------------------------------------------------------

def score_and_prepare(
    df: pd.DataFrame,
    X: pd.DataFrame,
    pipeline: Pipeline,
    cfg: dict,
    feature_cols: list
) -> pd.DataFrame:
    """Score the dataset and prepare a DataFrame for staging/upsert."""
    log.info("Scoring dataset")
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)[:, 1]
    else:
        probabilities = pipeline.predict(X)

    predictions = (probabilities >= cfg["probability_threshold"]).astype(int)

    df["predicted_recall_probability"] = probabilities
    df["predicted_recall_flag"] = predictions
    df["model_version"] = os.path.basename(cfg["model_path"])
    df["scoring_run_date"] = datetime.utcnow()

    out_df = pd.DataFrame({
        "PMA_PMN_NUM": df["PMA_PMN_NUM"],
        "model_version": df["model_version"],
        "predicted_recall_flag": df["predicted_recall_flag"].astype(int),
        "predicted_recall_probability": df["predicted_recall_probability"].astype(float),
        "scoring_run_date": df["scoring_run_date"]
    })

    snapshot_list = []
    for _, row in df.iterrows():
        snapshot = {c: row.get(c) for c in feature_cols}
        snapshot_list.append(json.dumps(snapshot, default=str))

    out_df["model_input_snapshot"] = snapshot_list
    log.info("Prepared output for %d rows", len(out_df))
    return out_df

# ------------------------------------------------------------------------------
# Step 6: Write to SQL and upsert
# ------------------------------------------------------------------------------

def write_to_sql(df: pd.DataFrame, engine, cfg: dict) -> None:
    """
    Writes the model scoring output to SQL Server in a reliable, resumable way.
    1. Creates a staging table.
    2. Deduplicates the staging data.
    3. Merges results into the target table using an UPSERT pattern.
    """

    staging_table = cfg.get("staging_table", "model.device_risk_scores_staging")
    target_table = cfg.get("target_table", "model.device_risk_scores")

    log.info("Starting write_to_sql process.")
    log.info("Writing staging table: %s", staging_table)

    # Step 1 ─ Write to staging table
    try:
        df.to_sql(
            name=staging_table.split(".")[-1],
            schema=staging_table.split(".")[0],
            con=engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=10000,
        )
        log.info("Staging table written successfully with %d rows.", len(df))
    except Exception as exc:
        log.error("Failed to write to staging table: %s", exc)
        raise

    # Step 2 ─ Deduplicate within the staging table
    try:
        dedup_sql = f"""
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY PMA_PMN_NUM
                       ORDER BY scoring_run_date DESC
                   ) AS rn
            FROM {staging_table}
        )
        DELETE FROM ranked WHERE rn > 1;
        """
        with engine.begin() as conn:
            conn.execute(text(dedup_sql))
        log.info("Deduplication complete in staging table.")
    except Exception as exc:
        log.warning("Deduplication skipped or failed: %s", exc)

    # Step 3 ─ Merge into target table (UPSERT)
    merge_sql = f"""
    MERGE {target_table} AS target
    USING {staging_table} AS src
      ON target.PMA_PMN_NUM = src.PMA_PMN_NUM
    WHEN MATCHED THEN
      UPDATE SET
          target.model_version = src.model_version,
          target.predicted_recall_flag = src.predicted_recall_flag,
          target.predicted_recall_probability = src.predicted_recall_probability,
          target.scoring_run_date = src.scoring_run_date,
          target.model_input_snapshot = src.model_input_snapshot
    WHEN NOT MATCHED BY TARGET THEN
      INSERT (PMA_PMN_NUM, model_version, predicted_recall_flag,
              predicted_recall_probability, scoring_run_date, model_input_snapshot)
      VALUES (src.PMA_PMN_NUM, src.model_version, src.predicted_recall_flag,
              src.predicted_recall_probability, src.scoring_run_date, src.model_input_snapshot);
    """

    try:
        with engine.begin() as conn:
            conn.execute(text(merge_sql))
        log.info("Merge completed successfully into %s.", target_table)
    except Exception as exc:
        log.error("Error during merge operation: %s", exc)
        raise

    log.info("write_to_sql process finished successfully.")


# ------------------------------------------------------------------------------
# Step 7: Save model and feature importance
# ------------------------------------------------------------------------------

def save_model_and_importance(
    pipeline: Pipeline, X: pd.DataFrame, engine, cfg: dict
):
    """Save trained model to disk and write feature importance to SQL."""
    joblib.dump(pipeline, cfg["model_path"])
    log.info("Model saved to: %s", cfg["model_path"])

    try:
        importances = pipeline.named_steps["clf"].feature_importances_
        feat_names = X.columns
        fi_df = pd.DataFrame({
            "feature": feat_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        fi_df.to_sql(cfg["feature_importance_table"],
                     con=engine, if_exists="replace", index=False)
        log.info("Feature importances written to table: %s",
                 cfg["feature_importance_table"])
    except Exception as err:
        log.warning("Could not compute feature importances: %s", err)

# ------------------------------------------------------------------------------
# Main orchestration
# ------------------------------------------------------------------------------

def main():
    """Main routine for extraction, feature engineering, training/scoring and loading."""
    cfg = CONFIG
    engine = get_engine(cfg)

    df = extract_view(engine, cfg["view_name"])

    if "PMA_PMN_NUM" not in df.columns:
        alt = [c for c in df.columns if c.lower() == "pma_pmn_num"]
        if alt:
            df.rename(columns={alt[0]: "PMA_PMN_NUM"}, inplace=True)
        else:
            raise RuntimeError(
                "ERROR: required column 'PMA_PMN_NUM' not found in view."
            )

    df, X, feat_cols = feature_engineer(df)
    df = create_label(df)
    y = df["recall_within_5yrs"].astype(int)

    if cfg["train_model"]:
        model = train_model(X, y, cfg)
        save_model_and_importance(model, X, engine, cfg)
    else:
        if not os.path.exists(cfg["model_path"]):
            raise RuntimeError(
                f"Model file not found: {cfg['model_path']}"
            )
        model = joblib.load(cfg["model_path"])
        log.info("Loaded existing model from %s", cfg["model_path"])

    out_df = score_and_prepare(df, X, model, cfg, feat_cols)
    write_to_sql(out_df, engine, cfg)

    log.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
