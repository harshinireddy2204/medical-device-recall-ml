import os
from datetime import datetime
import pandas as pd
from joblib import dump, load
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import sys
import joblib
import numpy as np
from pathlib import Path
import logging 

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
DATA_DIR = r"C:\Users\harsh\FDA_pipeline"
RAW_PATH = os.path.join(DATA_DIR, "vw_fda_device_integrated.parquet")
FEAT_PATH = os.path.join(DATA_DIR, "features.parquet")
PRED_PATH = os.path.join(DATA_DIR, "predictions.parquet")
MODEL_PATH = os.path.join(DATA_DIR, "rf_model_v1.joblib")

SQL_CONN_STR = (
    "mssql+pyodbc:///?odbc_connect="
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=FDADatabase;"
    "Trusted_Connection=yes;"
)
engine = create_engine(SQL_CONN_STR)

cfg = {
    "staging_table": "model.device_risk_scores_staging",
    "target_table": "model.device_risk_scores",
}

# ---------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------
def setup_logging(log_file: str = "fda_pipeline.log"):
    """
    Configure logging to write to file and console safely with UTF-8 encoding.
    """
    log = logging.getLogger()
    log.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    if log.hasHandlers():
        log.handlers.clear()

    # File handler (UTF-8)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    log.addHandler(file_handler)

    # Console handler (UTF-8)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(file_formatter)
    if hasattr(console_handler.stream, "reconfigure"):
        console_handler.stream.reconfigure(encoding="utf-8")
    log.addHandler(console_handler)

    log.info("=== Logging initialized (UTF-8 safe) ===")
    return log

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def ensure_file(path: str, description: str, log) -> bool:
    exists = os.path.exists(path)
    log.info("%s exists: %s", description, exists)
    return exists

# ---------------------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------------------
def load_extracted_data(log) -> pd.DataFrame:
    if ensure_file(RAW_PATH, "Raw parquet", log):
        log.info("Loading cached extraction from %s", RAW_PATH)
        return pd.read_parquet(RAW_PATH)
    else:
        raise FileNotFoundError("vw_fda_device_integrated.parquet not found! Rerun extraction first.")

# ---------------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING
# ---------------------------------------------------------------------
def feature_engineer_resumable(df: pd.DataFrame, log) -> pd.DataFrame:
    if ensure_file(FEAT_PATH, "Feature parquet", log):
        log.info("Loading cached features.")
        return pd.read_parquet(FEAT_PATH)

    log.info("Generating new engineered features.")
    df["time_on_market_years"] = (
        (pd.to_datetime("today") - pd.to_datetime(df["first_event_date"]))
        .dt.days / 365.25
    ).fillna(0.0)
    df["has_recall"] = df["recall_status"].notna().astype(int)
    df["event_density"] = df["adverse_event_count"] / df["unique_manufacturers"].clip(lower=1)
    df.to_parquet(FEAT_PATH, compression="snappy")
    log.info("Features saved to %s", FEAT_PATH)
    return df

# ---------------------------------------------------------------------
# STEP 3: TRAIN OR LOAD MODEL
# ---------------------------------------------------------------------


def train_or_load_model(df_feat: pd.DataFrame):
    model_path = os.path.join(os.getcwd(), "rf_model_v3.joblib")

    if os.path.exists(model_path):
        logging.info("Model file exists: True")
        logging.info(f"Loading existing model from {model_path}")
        return joblib.load(model_path)

    target_col = "recall_flag" if "recall_flag" in df_feat.columns else "has_recall"
    if target_col not in df_feat.columns:
        raise ValueError("No recall_flag or has_recall column found in features!")

    logging.info(f"Using target column: {target_col}")

    df_feat = df_feat.replace(["-", "NA", "N/A", "None", ""], np.nan)
    drop_cols = [
        col for col in df_feat.columns
        if df_feat[col].nunique() <= 1 or col.lower().endswith("_id")
    ]
    if drop_cols:
        logging.info(f"Dropping low-variance columns: {drop_cols}")
        df_feat = df_feat.drop(columns=drop_cols)
 
    X = df_feat.drop(columns=[target_col])
    y = df_feat[target_col].astype(int)

    # --- Encode categoricals efficiently ---
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    logging.info(f"Label-encoding {len(cat_cols)} categorical columns, keeping {len(numeric_cols)} numeric columns.")

    for col in cat_cols:
        le = LabelEncoder()
        X[col] = X[col].astype(str).fillna("missing")
        try:
            X[col] = le.fit_transform(X[col])
        except Exception as e:
            logging.warning(f"Label encoding failed for {col}: {e}")
            X[col] = 0

    X = X.fillna(0)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    logging.info("Training RandomForest model with Label Encoded features...")
    calibrated_rf = CalibratedClassifierCV(
        estimator=rf,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    )
    calibrated_rf.fit(X, y)

    cv_scores = cross_val_score(calibrated_rf, X, y, cv=3, scoring="roc_auc")
    logging.info(f"Calibrated Model CV ROC-AUC: {np.mean(cv_scores):.3f}")

    joblib.dump(calibrated_rf, model_path)
    logging.info(f"Saved model to {model_path}")

    return calibrated_rf

def score_and_save(df_feat: pd.DataFrame, model, log) -> pd.DataFrame:
    """
    Score the dataset using the trained model and return SQL-ready predictions.

    Features:
      ✓ Cleans text fields (quotes, spaces, brackets, etc.)
      ✓ Converts all non-numeric columns to category codes
      ✓ Aligns columns with the model’s training feature list
      ✓ Predicts calibrated probability and risk category
      ✓ Returns only the SQL-ready columns
    """

    log.info("Scoring dataset...")

    # ---------------------------------------------------
    # 1. CLEAN AND NORMALIZE ALL INPUT FEATURES
    # ---------------------------------------------------
    X = df_feat.copy()

    for col in X.columns:
        if X[col].dtype == "object" or pd.api.types.is_string_dtype(X[col]):
            X[col] = (
                X[col]
                .astype(str)
                .str.strip()
                .str.replace('"', '', regex=False)
                .str.replace("'", "", regex=False)
                .str.replace(r"[\[\]\(\)`]", "", regex=True)
                .replace(["-", "NA", "N/A", "None", "nan", ""], "missing")
            )

    # Fix infinities or NaNs
    X = X.replace([-np.inf, np.inf], np.nan).fillna("missing")

    # ---------------------------------------------------
    # 2. ENCODE ALL NON-NUMERIC COLUMNS TO INTEGERS
    # ---------------------------------------------------
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = X[col].astype("category").cat.codes

    X = X.fillna(0)

    # ---------------------------------------------------
    # 3. ALIGN FEATURES WITH TRAINED MODEL
    # ---------------------------------------------------
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)

        missing_cols = [c for c in expected_cols if c not in X.columns]
        extra_cols   = [c for c in X.columns if c not in expected_cols]

        if missing_cols:
            log.warning("Adding missing model columns with zeros: %s", missing_cols)
            for col in missing_cols:
                X[col] = 0

        if extra_cols:
            log.warning("Dropping unused extra columns: %s", extra_cols)

        # Final aligned set
        X = X[expected_cols]

    # ---------------------------------------------------
    # 4. RUN PREDICTIONS
    # ---------------------------------------------------
    try:
        df_feat["predicted_recall_probability"] = model.predict_proba(X)[:, 1]
    except Exception as e:
        log.error("Prediction error: %s", str(e))
        raise

    # ---------------------------------------------------
    # 5. ASSIGN RISK CATEGORY
    # ---------------------------------------------------
    def categorize(prob):
        if prob <= 0.25:
            return "Low"
        elif prob < 0.50:
            return "Medium"
        elif prob < 0.75:
            return "High"
        else:
            return "Very High"

    df_feat["risk_category"] = df_feat["predicted_recall_probability"].apply(categorize)

    # ---------------------------------------------------
    # 6. ADD METADATA
    # ---------------------------------------------------
    df_feat["model_version"] = "v3.0"
    df_feat["predicted_recall_flag"] = (df_feat["predicted_recall_probability"] > 0.5).astype(int)
    df_feat["scoring_run_date"] = pd.Timestamp.now()

    # ---------------------------------------------------
    # 7. SAVE LOCAL PARQUET (OPTIONAL)
    # ---------------------------------------------------
    out_path = os.path.join(os.getcwd(), "predictions.parquet")
    df_feat.to_parquet(out_path, index=False)
    log.info("Saved predictions to %s", out_path)

    # ---------------------------------------------------
    # 8. PREP SQL-READY OUTPUT
    # ---------------------------------------------------
    df_sql = df_feat[
        [
            "PMA_PMN_NUM",
            "model_version",
            "predicted_recall_flag",
            "predicted_recall_probability",
            "scoring_run_date",
        ]
    ].copy()

    df_sql = df_sql.drop_duplicates(subset=["PMA_PMN_NUM"], keep="last")

    log.info("Returning %d SQL-ready rows.", len(df_sql))
    return df_sql



# ---------------------------------------------------------------------
# STEP 5: WRITE TO SQL (DEDUPLICATED)
# ---------------------------------------------------------------------
def write_to_sql(df: pd.DataFrame, engine, cfg: dict, log):
    """
    Writes FDA model predictions to SQL Server safely.
    
    Features:
      ✓ Cleans PMA/PMN IDs (extracts K123456 / P100061)
      ✓ Drops invalid IDs
      ✓ Deduplicates by PMA_PMN_NUM
      ✓ Creates staging schema if needed
      ✓ Uses fast_executemany for FAST + NO-FREEZE inserts
      ✓ MERGE upserts into target table
    """

    import re
    from sqlalchemy import text

    staging = cfg["staging_table"]
    target = cfg["target_table"]
    staging_schema, staging_table = staging.split(".")
    target_schema, target_table = target.split(".")

    # --------------------------------------------
    # 1. Validate required columns
    # --------------------------------------------
    required_cols = [
        "PMA_PMN_NUM",
        "model_version",
        "predicted_recall_flag",
        "predicted_recall_probability",
        "scoring_run_date",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_sql = df[required_cols].copy()

    # --------------------------------------------
    # 2. CLEAN / EXTRACT REAL PMA / PMN NUMBERS
    # --------------------------------------------
    def extract_valid_id(value):
        """Extracts K123456 or P100061 from any messy string."""
        if pd.isna(value):
            return np.nan

        value = str(value).strip()
        value = re.sub(r'["`\[\]\': ]', '', value)

        # Match K123456 or P123456 format
        m = re.search(r'(K\d{6}|P\d{6})', value, re.IGNORECASE)
        return m.group(1).upper() if m else np.nan

    df_sql["PMA_PMN_NUM"] = df_sql["PMA_PMN_NUM"].apply(extract_valid_id)

    before = len(df_sql)
    df_sql = df_sql.dropna(subset=["PMA_PMN_NUM"])
    after = len(df_sql)
    log.info("Filtered out %d invalid PMA/PMN rows.", before - after)

    # --------------------------------------------
    # 3. Deduplicate latest scoring_run_date
    # --------------------------------------------
    if df_sql.duplicated(subset=["PMA_PMN_NUM"]).any():
        dup = df_sql[df_sql.duplicated(subset=["PMA_PMN_NUM"], keep=False)]
        log.warning("Found %d duplicates — keeping most recent.", len(dup))

        df_sql = (
            df_sql.sort_values("scoring_run_date")
                  .drop_duplicates(subset=["PMA_PMN_NUM"], keep="last")
        )

    # --------------------------------------------
    # 4. Ensure staging schema exists
    # --------------------------------------------
    with engine.begin() as conn:
        log.info("Ensuring schema [%s] exists...", staging_schema)
        conn.execute(
            text(
                f"""
                IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = '{staging_schema}')
                EXEC('CREATE SCHEMA {staging_schema}');
                """
            )
        )

        # --------------------------------------------
        # 5. DROP staging table if exists
        # --------------------------------------------
        conn.execute(
            text(
                f"""
                IF OBJECT_ID('{staging_schema}.{staging_table}', 'U') IS NOT NULL
                    DROP TABLE {staging_schema}.{staging_table};
                """
            )
        )

    # --------------------------------------------
    # 6. SAFE INSERT — NO FREEZE USING fast_executemany
    # --------------------------------------------
    log.info("Writing %d cleaned rows to staging table [%s].[%s] (fast mode)",
             len(df_sql), staging_schema, staging_table)

    # Clean bad whitespace
    for col in df_sql.select_dtypes(include=["object"]).columns:
        df_sql[col] = (
            df_sql[col]
            .astype(str)
            .str.replace(r"[\r\n\t]", " ", regex=True)
            .str.strip()
        )

    # --- USE fast_executemany (NO multi-insert) ---
    from sqlalchemy import event
    import pyodbc

    @event.listens_for(engine, "before_cursor_execute")
    def receive_before_cursor_execute(conn, cursor, stmt, params, context, executemany):
        if executemany:
            cursor.fast_executemany = True

    df_sql.to_sql(
        name=staging_table,
        schema=staging_schema,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=2000,   # safe batch size
        method=None,       # NO multi → prevents freezing
    )

    # --------------------------------------------
    # 7. MERGE UPSERT
    # --------------------------------------------
    log.info("Running MERGE into target table [%s].[%s]", target_schema, target_table)

    merge_sql = f"""
    MERGE [{target_schema}].[{target_table}] AS target
    USING [{staging_schema}].[{staging_table}] AS src
        ON target.[PMA_PMN_NUM] = src.[PMA_PMN_NUM]
    WHEN MATCHED THEN
        UPDATE SET
            target.[model_version]               = src.[model_version],
            target.[predicted_recall_flag]        = src.[predicted_recall_flag],
            target.[predicted_recall_probability] = src.[predicted_recall_probability],
            target.[scoring_run_date]             = src.[scoring_run_date]
    WHEN NOT MATCHED BY TARGET THEN
        INSERT (
            [PMA_PMN_NUM],
            [model_version],
            [predicted_recall_flag],
            [predicted_recall_probability],
            [scoring_run_date]
        )
        VALUES (
            src.[PMA_PMN_NUM],
            src.[model_version],
            src.[predicted_recall_flag],
            src.[predicted_recall_probability],
            src.[scoring_run_date]
        );
    """

    with engine.begin() as conn:
        conn.execute(text(merge_sql))

    log.info(
        "✅ Merge complete — %d rows upserted into [%s].[%s].",
        len(df_sql),
        target_schema,
        target_table,
    )


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def main():
    # Initialize logging
    log = setup_logging("fda_pipeline.log")
    log.info("=== FDA DEVICE PIPELINE STARTED ===")

 

    # Step 1: Load extracted or cached data
    df_raw = load_extracted_data(log)

    # Step 2: Feature engineering (resumable)
    df_feat = feature_engineer_resumable(df_raw, log)

    # Step 3: Train or load model
    model = train_or_load_model(df_feat)

    # Step 4: Score dataset and prepare predictions
    df_pred = score_and_save(df_feat, model, log)

    # Step 5: Write results to SQL
    write_to_sql(df_pred, engine, cfg, log)

    log.info("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()

