import pyodbc
import pandas as pd
import matplotlib.pyplot as plt

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=FDADatabase;"
    "Trusted_Connection=yes;"
)

query = """
SELECT
    root_cause_description,
    COUNT(*) AS device_count,
    AVG(rpss) AS avg_rpss
FROM model.device_rpss
WHERE rpss_category IN ('High', 'Critical')
  AND root_cause_description IS NOT NULL
GROUP BY root_cause_description
ORDER BY device_count DESC
"""

df = pd.read_sql(query, conn).head(10)

plt.figure(figsize=(10,6))
bars = plt.barh(
    df["root_cause_description"],
    df["device_count"],
    color="#e67e22"
)

plt.title("Top Root Causes Driving High & Critical RPSS Devices", fontsize=14, weight="bold")
plt.xlabel("Number of High-Risk Devices")
plt.ylabel("Recall Root Cause")
plt.gca().invert_yaxis()
plt.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()
