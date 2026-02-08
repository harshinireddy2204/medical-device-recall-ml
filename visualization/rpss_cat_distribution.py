import pyodbc
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Database Connection
# -------------------------------
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=FDADatabase;"
    "Trusted_Connection=yes;"
)

# Chart 2: RPSS Category Distribution
# -------------------------------
query_cat = """
SELECT
    device_class,
    rpss_category,
    COUNT(*) AS device_count
FROM model.device_rpss
WHERE device_class IS NOT NULL
GROUP BY device_class, rpss_category
"""

df_cat = pd.read_sql(query_cat, conn)

pivot = df_cat.pivot(
    index="device_class",
    columns="rpss_category",
    values="device_count"
).fillna(0)

pivot.plot(kind="bar", stacked=True)
plt.xlabel("Device Class")
plt.ylabel("Number of Devices")
plt.title("RPSS Risk Category Distribution by Device Class")
plt.tight_layout()
plt.show()