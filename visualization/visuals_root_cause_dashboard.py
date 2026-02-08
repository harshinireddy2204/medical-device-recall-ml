import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# SQL connection
engine = create_engine(
    "mssql+pyodbc:///?odbc_connect="
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=FDADatabase;"
    "Trusted_Connection=yes;"
)

query = """
WITH high_risk AS (
    SELECT
        root_cause_description,
        device_class,
        rpss
    FROM model.device_rpss
    WHERE
        rpss_category IN ('High', 'Critical')
        AND root_cause_description IS NOT NULL
        AND device_class IS NOT NULL
)
SELECT
    root_cause_description,
    device_class,
    COUNT(*) AS device_count,
    AVG(rpss) AS avg_rpss
FROM high_risk
GROUP BY root_cause_description, device_class;
"""

df = pd.read_sql(query, engine)
# total devices per root cause
top_causes = (
    df.groupby("root_cause_description")["device_count"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
      .index
)

df = df[df["root_cause_description"].isin(top_causes)]

pivot_counts = df.pivot_table(
    index="root_cause_description",
    columns="device_class",
    values="device_count",
    aggfunc="sum",
    fill_value=0
)

avg_rpss = (
    df.groupby("root_cause_description")["avg_rpss"]
      .mean()
)
fig, ax1 = plt.subplots(figsize=(14, 7))

# Stacked bars
bottom = None
for cls in sorted(pivot_counts.columns):
    ax1.bar(
        pivot_counts.index,
        pivot_counts[cls],
        bottom=bottom,
        label=f"Class {int(cls)}"
    )
    bottom = pivot_counts[cls] if bottom is None else bottom + pivot_counts[cls]

ax1.set_ylabel("Number of High / Critical Risk Devices")
ax1.set_title("Top Recall Root Causes Driving High-Risk Medical Devices")
ax1.tick_params(axis='x', rotation=45, ha='right')
ax1.legend(title="Device Class")

# Second axis: RPSS severity
ax2 = ax1.twinx()
ax2.plot(
    avg_rpss.index,
    avg_rpss.values,
    marker="o"
)
ax2.set_ylabel("Average RPSS")

plt.tight_layout()
plt.show()
