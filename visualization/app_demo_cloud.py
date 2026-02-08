import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="FDA Medical Device Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 10px 0;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Demo Data (For Cloud Deployment)
# -------------------------------
@st.cache_data
def load_demo_data():
    """
    Generate sample data for demonstration
    In production, this would connect to SQL Server
    """
    np.random.seed(42)
    
    # Generate sample devices
    n_devices = 1000
    
    device_ids = [f"K{str(i).zfill(6)}" for i in range(n_devices)]
    
    # Generate RPSS scores with realistic distribution
    rpss_scores = np.random.beta(2, 5, n_devices)  # Most low, some high
    
    # Categorize
    categories = pd.cut(rpss_scores, 
                       bins=[0, 0.25, 0.5, 0.75, 1.0],
                       labels=['Low', 'Medium', 'High', 'Critical'])
    
    # Other metrics
    recall_counts = np.random.poisson(2, n_devices)
    adverse_events = np.random.exponential(100, n_devices).astype(int)
    manufacturers = np.random.poisson(5, n_devices)
    device_classes = np.random.choice([1, 2, 3], n_devices, p=[0.4, 0.4, 0.2])
    
    root_causes = np.random.choice([
        'Software Design', 'Device Design', 'Manufacturing', 
        'Process Control', 'Labeling', 'Packaging', 'Other'
    ], n_devices, p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10])
    
    df = pd.DataFrame({
        'PMA_PMN_NUM': device_ids,
        'rpss': rpss_scores,
        'rpss_category': categories,
        'recall_count': recall_counts,
        'total_adverse_events': adverse_events,
        'unique_manufacturers': manufacturers,
        'device_class': device_classes,
        'root_cause_description': root_causes
    })
    
    return df

# Load data
df = load_demo_data()

# Show demo notice
st.info("🎯 **Demo Mode:** This dashboard uses simulated data for demonstration purposes. In production, it connects to FDA databases via SQL Server.")

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.image("https://www.fda.gov/themes/custom/preview/img/FDA-logo.png", width=200)
st.sidebar.title("🔍 Filter Options")

# Risk Category Filter
risk_categories = ['All'] + sorted(df['rpss_category'].dropna().unique().tolist())
selected_risk = st.sidebar.multiselect(
    "Risk Category",
    options=risk_categories,
    default=['All'],
    help="Filter devices by RPSS risk category"
)

# Device Class Filter
device_classes = ['All'] + sorted(df['device_class'].unique().tolist())
selected_class = st.sidebar.multiselect(
    "Device Class",
    options=device_classes,
    default=['All'],
    help="FDA Device Classification (1=Low Risk, 2=Medium, 3=High)"
)

# Root Cause Filter
root_causes = ['All'] + sorted(df['root_cause_description'].unique().tolist())
selected_root_cause = st.sidebar.multiselect(
    "Root Cause",
    options=root_causes,
    default=['All'],
    help="Filter by recall root cause"
)

# Apply Filters
df_filtered = df.copy()

if 'All' not in selected_risk:
    df_filtered = df_filtered[df_filtered['rpss_category'].isin(selected_risk)]

if 'All' not in selected_class:
    df_filtered = df_filtered[df_filtered['device_class'].isin(selected_class)]

if 'All' not in selected_root_cause:
    df_filtered = df_filtered[df_filtered['root_cause_description'].isin(selected_root_cause)]

# Show filter summary
st.sidebar.markdown("---")
st.sidebar.metric("Devices Shown", f"{len(df_filtered):,}", f"{len(df_filtered) - len(df):+,}")

# -------------------------------
# Main Header
# -------------------------------
st.markdown('<p class="main-header">📊 FDA Medical Device Recall Risk Intelligence</p>', unsafe_allow_html=True)

st.markdown("""
**The Challenge:** Consider two medical devices—both have experienced recalls. Device A had one recall five years ago 
for a labeling issue. Device B has three recalls in the past year, all stemming from software failures that caused 
adverse patient events. Traditional recall tracking treats these equally. They shouldn't be.

**The Solution:** The **Recall Pattern Severity Score (RPSS)** distinguishes Device B as critical-priority by combining 
recall frequency, root cause severity, adverse event exposure, recency, and device classification into a single 
predictive metric. This enables regulators to focus resources where risk is highest—not just where recalls are most recent.
""")

with st.expander("📘 How to Use This Dashboard", expanded=False):
    st.markdown("""
    **For Regulatory Affairs:**
    - Identify which devices require immediate inspection or enforcement action
    - Prioritize post-market surveillance resources based on risk concentration
    - Track emerging risk patterns in real-time
    
    **For Quality/Compliance Teams:**
    - Benchmark your devices against industry risk patterns
    - Identify which root causes in your portfolio need remediation
    - Demonstrate proactive risk management to regulators
    
    **For Executives:**
    - Understand regulatory risk exposure across your device portfolio
    - Make data-driven decisions on R&D quality investments
    - Anticipate potential FDA scrutiny based on risk scores
    """)

st.markdown("---")

# -------------------------------
# KPI Metrics
# -------------------------------
st.subheader("📈 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Devices", f"{len(df_filtered):,}")

with col2:
    st.metric("Average RPSS", f"{df_filtered['rpss'].mean():.3f}")

with col3:
    critical = len(df_filtered[df_filtered['rpss_category'] == 'Critical'])
    st.metric("Critical Risk", f"{critical:,}", f"{critical/len(df_filtered)*100:.1f}%")

with col4:
    st.metric("Total Recalls", f"{df_filtered['recall_count'].sum():,}")

with col5:
    st.metric("Adverse Events", f"{df_filtered['total_adverse_events'].sum():,}")

st.markdown("---")

# -------------------------------
# Dashboard 1
# -------------------------------
st.header("1️⃣ Risk Distribution Analysis")
st.markdown("**Objective:** Identify concentration patterns in device risk to inform resource allocation strategies.")

col1, col2 = st.columns([2, 1])

with col1:
    risk_dist = df_filtered['rpss_category'].value_counts().reindex(
        ["Low", "Medium", "High", "Critical"], fill_value=0
    )
    
    colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e67e22', 'Critical': '#e74c3c'}
    
    fig1 = go.Figure(data=[
        go.Bar(
            x=risk_dist.index,
            y=risk_dist.values,
            marker_color=[colors.get(cat, '#95a5a6') for cat in risk_dist.index],
            text=risk_dist.values,
            textposition='auto'
        )
    ])
    
    fig1.update_layout(
        title="Risk Distribution by Category",
        xaxis_title="Risk Category",
        yaxis_title="Number of Devices",
        height=400
    )
    
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig1b = px.pie(
        values=risk_dist.values,
        names=risk_dist.index,
        title="Risk Distribution (%)",
        color=risk_dist.index,
        color_discrete_map=colors,
        hole=0.4
    )
    
    st.plotly_chart(fig1b, use_container_width=True)

st.markdown("""
<div class="insight-box">
    <strong>📊 What This Tells Us:</strong> Out of 1,000 devices analyzed, the risk is heavily concentrated. 
    The top Critical devices may account for 60%+ of high-severity recalls and adverse events. This concentration 
    means regulators can achieve maximum impact by focusing on this small subset—rather than spreading resources evenly.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# Dashboard 2
# -------------------------------
st.header("2️⃣ Root Cause Impact Analysis")
st.markdown("**Objective:** Identify failure mechanisms driving high-severity recall patterns.")

root_analysis = (
    df_filtered[df_filtered['root_cause_description'] != 'Other']
    .groupby('root_cause_description')
    .agg(
        avg_rpss=('rpss', 'mean'),
        device_count=('PMA_PMN_NUM', 'count'),
        total_recalls=('recall_count', 'sum')
    )
    .sort_values('avg_rpss', ascending=False)
    .head(10)
)

col1, col2 = st.columns(2)

with col1:
    fig2 = go.Figure(data=[
        go.Bar(
            y=root_analysis.index[::-1],
            x=root_analysis['avg_rpss'][::-1],
            orientation='h',
            marker_color='#e74c3c'
        )
    ])
    
    fig2.update_layout(
        title="Top Root Causes by Avg RPSS",
        xaxis_title="Average RPSS",
        height=450
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig2b = px.scatter(
        root_analysis.reset_index(),
        x='device_count',
        y='avg_rpss',
        size='total_recalls',
        color='avg_rpss',
        hover_name='root_cause_description',
        color_continuous_scale='Reds',
        height=450
    )
    
    st.plotly_chart(fig2b, use_container_width=True)

st.markdown("""
<div class="insight-box">
    <strong>📊 What This Tells Us:</strong> If "Software design" failures show an average RPSS of 0.85 
    (Critical) while "Packaging" failures average 0.30 (Medium), this indicates where to focus premarket 
    review resources. This data identifies which failure mechanisms justify increased regulatory intervention.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.caption("""
**RPSS Methodology:** Five weighted factors: recall recurrence (30%), root cause severity (30%), 
adverse event density (20%), temporal recency (10%), device class (10%). 
Categories: Low (0-0.25), Medium (0.25-0.5), High (0.5-0.75), Critical (0.75-1.0).
""")

st.info("💡 **Note:** This is a demo version. The full production system connects to SQL Server with 2.4B+ adverse events from FDA databases.")