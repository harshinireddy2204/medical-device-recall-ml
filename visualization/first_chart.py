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
    device_class,
    AVG(rpss) AS avg_rpss
FROM model.device_rpss
GROUP BY device_class
"""

df = pd.read_sql(query, conn)

plt.bar(df["device_class"], df["avg_rpss"])
plt.xlabel("Device Class")
plt.ylabel("Average RPSS")
plt.title("Average RPSS by Device Class")
plt.show()
