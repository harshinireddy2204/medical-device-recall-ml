import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

conn = create_engine(
    "mssql+pyodbc:///?odbc_connect="
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=FDADatabase;"
    "Trusted_Connection=yes;"
)

query = """SELECT
    device_class,
    failure_category,
    COUNT(*) AS device_count,
    AVG(rpss) AS avg_rpss,
    (
        AVG(rpss)
        * COUNT(*)
        * device_class
    ) AS risk_amplification_score
FROM dbo.vw_device_rpss_categorized
WHERE
    rpss_category IN ('High', 'Critical')
    AND device_class IS NOT NULL
GROUP BY device_class, failure_category;

"""
df = pd.read_sql(query, conn)

plt.figure(figsize=(14, 7))

plt.scatter(
    df['failure_category'],
    df['device_class'],
    s=df['risk_amplification_score'] / 5,
    alpha=0.7
)

plt.xlabel('Failure Category')
plt.ylabel('Device Class')
plt.title('Risk Amplification Matrix (Severity × Exposure × Class)')
plt.xticks(rotation=45, ha='right')
plt.grid(True)

plt.tight_layout()
plt.show()
