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

# -------------------------------
# Dashboard 1: Average RPSS by Device Class
# -------------------------------
query = """
SELECT
    device_class,
    AVG(rpss) AS avg_rpss
FROM model.device_rpss
WHERE device_class IS NOT NULL
GROUP BY device_class
ORDER BY device_class
"""

df = pd.read_sql(query, conn)

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(8,5))

bars = plt.bar(
    df["device_class"].astype(str),
    df["avg_rpss"],
    color=["#7f8c8d", "#f39c12", "#c0392b"]  # Class 1, 2, 3
)

plt.title("Average RPSS by FDA Device Class", fontsize=14, weight="bold")
plt.xlabel("FDA Device Class")
plt.ylabel("Average RPSS")
plt.ylim(0, 1)

# Value labels
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{bar.get_height():.2f}",
        ha="center",
        fontsize=10
    )

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
