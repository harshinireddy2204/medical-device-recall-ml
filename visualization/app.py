import streamlit as st
import pyodbc
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="FDA Medical Device Risk Intelligence",
    layout="wide"
)

st.title("FDA Medical Device Recall Risk Intelligence (RPSS)")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=FDADatabase;"
        "Trusted_Connection=yes;"
    )
    return pd.read_sql(
        "SELECT * FROM dbo.vw_device_rpss_categorized",
        conn
    )

df = load_data()

# ===============================
# DASHBOARD 1 — OVERVIEW
# ===============================
st.header("1️⃣ Overall Risk Landscape")

risk_dist = df["rpss_category"].value_counts().reindex(
    ["Low", "Medium", "High", "Critical"]
)

st.bar_chart(risk_dist)

st.caption(
    "🔍 **Key Takeaway:** Most devices are low to medium risk, "
    "but a critical high-risk tail exists that drives regulatory concern."
)

# ===============================
# DASHBOARD 2 — ROOT CAUSE DRIVERS (FIXED)
# ===============================
st.header("2️⃣ Top Root Causes Driving Risk")

root_cause = (
    df[df["root_cause_description"] != "Other"]
    .groupby("root_cause_description")
    .agg(
        avg_rpss=("rpss", "mean"),
        device_count=("PMA_PMN_NUM", "nunique")
    )
    .query("device_count >= 10")
    .sort_values("avg_rpss", ascending=False)
    .head(10)
)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.barh(
    root_cause.index[::-1],
    root_cause["avg_rpss"][::-1]
)
ax2.set_xlabel("Average RPSS")
ax2.set_ylabel("Root Cause")
ax2.set_title("Highest-Risk Recall Root Causes")

st.pyplot(fig2)

st.caption(
    "🔍 **Key Takeaway:** Software, design, and manufacturing failures consistently "
    "produce the most severe recall patterns."
)

# ===============================
# DASHBOARD 3 — TOP RISK DEVICES (FIXED)
# ===============================
st.header("3️⃣ Devices Requiring Immediate Attention")

top_devices = (
    df[df["root_cause_description"] != "Other"]
    .sort_values("rpss", ascending=False)
    .loc[:, [
        "PMA_PMN_NUM",
        "rpss",
        "rpss_category",
        "recall_count",
        "device_class",
        "root_cause_description"
    ]]
    .head(20)
)

st.dataframe(
    top_devices.style
    .background_gradient(subset=["rpss"], cmap="Reds"),
    use_container_width=True
)

st.caption(
    "🔍 **Key Takeaway:** Regulatory risk is highly concentrated — a small number of "
    "devices account for a disproportionate share of concern."
)

# ===============================
# DASHBOARD 4 — REGULATORY PRIORITY (REDESIGNED)
# ===============================
st.info(
    "🧠 **Why RPSS Exists**\n\n"
    "FDA recall data alone does not indicate which devices require immediate attention. "
    "Many devices experience recalls, but the *pattern* of those recalls — frequency, "
    "severity, exposure, recency, and inherent device risk — determines true concern.\n\n"
    "**RPSS consolidates these dimensions into a single score (0–1)**, enabling "
    "risk-based regulatory prioritization rather than binary recall tracking."
)

st.header("4️⃣ Regulatory Priority Ranking")

df["reg_priority"] = (
    df["rpss"] *
    df["recall_count"] *
    df["device_class"]
)

priority = (
    df[df["root_cause_description"] != "Other"]
    .groupby("root_cause_description")
    .agg(reg_priority=("reg_priority", "sum"))
    .sort_values("reg_priority", ascending=False)
    .head(10)
)

fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.hlines(
    y=priority.index[::-1],
    xmin=0,
    xmax=priority["reg_priority"][::-1]
)
ax4.plot(
    priority["reg_priority"][::-1],
    priority.index[::-1],
    "o"
)

ax4.set_xlabel("Regulatory Priority Index")
ax4.set_ylabel("Failure Mechanism")
ax4.set_title("Where Regulatory Attention Should Focus")

st.pyplot(fig4)

st.caption(
    "🔍 **Key Takeaway:** When severity, recurrence, and device class are combined, "
    "software and design-related failures emerge as top regulatory priorities."
)

# ===============================
# EXECUTIVE SUMMARY
# ===============================
st.divider()
st.subheader("📌 Executive Summary")

st.markdown("""
- RPSS translates fragmented FDA post-market data into an actionable risk signal  
- Risk is **not evenly distributed** — it concentrates around specific failure mechanisms  
- A small number of devices drive a large share of regulatory concern  
- Prioritization should focus on **repeat, high-severity failures in high-class devices**
""")
