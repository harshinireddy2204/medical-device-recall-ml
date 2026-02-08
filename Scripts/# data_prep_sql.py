# data_prep_sql.py
import pyodbc, pandas as pd, logging
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [INFO] %(message)s")

# SQL Connection (adjust your server/db info)
CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=FDADatabase;"
    "Trusted_Connection=yes;"
)

VIEW = "dbo.vw_FDA_Device_Integrated"
LOOKAHEAD_DAYS = 1825  # 5 years

def load_from_sql():
    logging.info(f"Loading data from SQL view: {VIEW}")
    query = f"SELECT * FROM {VIEW};"
    with pyodbc.connect(CONN_STR) as conn:
        df = pd.read_sql(query, conn, parse_dates=['first_event_date','last_event_date','event_date_initiated'])
    logging.info(f"Loaded {len(df):,} rows from SQL")
    return df

def build_labels(df):
    if 'PMA_PMN_NUM' not in df.columns:
        raise SystemExit("Missing PMA_PMN_NUM — ensure your view has it.")
    df['product_code'] = df['PMA_PMN_NUM']
    df['recall_risk'] = 0

    recall_df = df[df['recall_status'].isin(['Terminated','Open','Classified'])][
        ['product_code','event_date_initiated']
    ].dropna()

    def has_recall(row):
        prod, last = row['product_code'], row['last_event_date']
        if pd.isna(last): return 0
        window_end = last + timedelta(days=LOOKAHEAD_DAYS)
        mask = (recall_df['product_code'] == prod) & \
               (recall_df['event_date_initiated'] > last) & \
               (recall_df['event_date_initiated'] <= window_end)
        return int(mask.any())

    df['recall_risk'] = df.apply(has_recall, axis=1)
    logging.info(f"Labeling complete — {df['recall_risk'].sum()} recalls identified.")
    return df

def main():
    df = load_from_sql()
    df = build_labels(df)
    df.to_parquet(r"C:\Users\harsh\FDA_pipeline\data\processed\fda_model_input_sql.parquet", index=False)
    logging.info("Saved model input parquet.")

if __name__ == "__main__":
    main()
