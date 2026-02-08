import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

engine = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=FDADatabase;"
    "Trusted_Connection=yes;"
)



df = pd.read_sql("""
SELECT
    root_cause_description,
    device_class,
    COUNT(*) AS cnt
FROM model.device_rpss
WHERE rpss_category IN ('High', 'Critical')
GROUP BY root_cause_description, device_class
""", engine)


pivot = df.pivot(
    index="device_class",
    columns="root_cause_description",
    values="avg_rpss"
)

plt.figure(figsize=(12, 6))
sns.heatmap(pivot, cmap="Reds", annot=False)
plt.title("RPSS Heatmap: Device Class vs Recall Root Cause")
plt.xlabel("Root Cause")
plt.ylabel("Device Class")
plt.tight_layout()
plt.show()
