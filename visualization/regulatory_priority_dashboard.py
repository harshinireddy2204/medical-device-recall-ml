import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# ============================================================
# DATABASE CONNECTION
# ============================================================
engine = create_engine(
    "mssql+pyodbc:///?odbc_connect="
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=FDADatabase;"
    "Trusted_Connection=yes;"
)

query = """
SELECT
    failure_category,
    device_class,
    COUNT(*) AS device_count,
    AVG(rpss) AS avg_rpss,
    SUM(CASE WHEN rpss_category = 'Critical' THEN 1 ELSE 0 END) AS critical_count,
    (
        AVG(rpss)
        + (0.5 * SUM(CASE WHEN rpss_category = 'Critical' THEN 1 ELSE 0 END))
        + (0.1 * COUNT(*))
    ) AS regulatory_priority_index
FROM dbo.vw_device_rpss_categorized
WHERE
    rpss_category IN ('High', 'Critical')
GROUP BY failure_category, device_class
ORDER BY regulatory_priority_index DESC;

"""

df = pd.read_sql(query, engine)

# ============================================================
# THRESHOLDS (MEDIANS DEFINE QUADRANTS)
# ============================================================
top10 = df.sort_values(
    'regulatory_priority_index',
    ascending=False
).head(10)

plt.figure(figsize=(12, 6))

plt.barh(
    top10['failure_category'] + ' (Class ' + top10['device_class'].astype(str) + ')',
    top10['regulatory_priority_index']
)

plt.xlabel('Regulatory Priority Index')
plt.title('Top 10 FDA Regulatory Priorities')
plt.gca().invert_yaxis()
plt.grid(axis='x')

plt.tight_layout()
plt.show()
