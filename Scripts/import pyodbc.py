import urllib
from sqlalchemy import create_engine, text, event

SQL_SERVER = "localhost"
DATABASE = "FDADatabase"
ODBC_DRIVER = "ODBC Driver 17 for SQL Server"   # or "ODBC Driver 18 for SQL Server"

raw = (
    f"DRIVER={{{ODBC_DRIVER}}};"
    f"SERVER={SQL_SERVER};"
    f"DATABASE={DATABASE};"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)
params = urllib.parse.quote_plus(raw)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}", pool_pre_ping=True)

# Enable fast_executemany for pyodbc to speed bulk inserts:
@event.listens_for(engine, "before_cursor_execute")
def _fast_executemany(conn, cursor, statement, parameters, context, executemany):
    if executemany:
        try:
            cursor.fast_executemany = True
        except Exception:
            pass

# Quick test
with engine.connect() as conn:
    rows = conn.execute(text("SELECT name FROM sys.databases")).fetchall()
    print(rows)
