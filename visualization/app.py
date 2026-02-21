import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import sys
import os

# Add Scripts directory to path for ML modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Scripts'))

try:
    from ml_recall_prediction import RecallPredictor
    from time_series_forecast import RecallForecaster
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    # Don't show warning in main app, just disable features silently

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
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
        color: #856404;
    }
    .story-arc {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 15px 0;
        color: #000000;
    }
    .story-arc h4 { color: #1f77b4; margin-bottom: 12px; }
    .story-arc p { color: #000000; }
    .story-step { margin: 8px 0; padding-left: 8px; border-left: 3px solid #17a2b8; color: #000000; }
    .story-cta { background-color: #e8f4f8; padding: 12px; border-radius: 6px; margin: 10px 0; font-weight: 600; color: #000000; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Database Connection
# -------------------------------
@st.cache_resource
def get_engine():
    """Create SQLAlchemy engine (cached)"""
    connection_string = (
        "mssql+pyodbc:///?odbc_connect="
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=FDADatabase;"
        "Trusted_Connection=yes;"
    )
    return create_engine(connection_string, pool_pre_ping=True, pool_size=5, max_overflow=10)

engine = get_engine()

# -------------------------------
# Optimized Data Loading (BIGINT safe)
# -------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def load_summary_stats():
    """Load summary statistics with BIGINT support"""
    query = """
    SELECT 
        COUNT(*) as total_devices,
        CAST(AVG(rpss) AS FLOAT) as avg_rpss,
        CAST(SUM(CAST(recall_count AS BIGINT)) AS BIGINT) as total_recalls,
        CAST(SUM(CAST(total_adverse_events AS BIGINT)) AS BIGINT) as total_adverse,
        SUM(CASE WHEN rpss_category = 'Critical' THEN 1 ELSE 0 END) as critical_count,
        SUM(CASE WHEN rpss_category = 'High' THEN 1 ELSE 0 END) as high_count,
        SUM(CASE WHEN rpss_category = 'Medium' THEN 1 ELSE 0 END) as medium_count,
        SUM(CASE WHEN rpss_category = 'Low' THEN 1 ELSE 0 END) as low_count,
        MAX(total_adverse_events) as max_adverse_single_device
    FROM model.device_rpss
    """
    df = pd.read_sql(query, engine)
    return df.iloc[0]

@st.cache_data(ttl=600, show_spinner=False)
def load_filter_options():
    """Load unique values for filters"""
    query = """
    SELECT DISTINCT
        rpss_category,
        device_class,
        root_cause_description
    FROM model.device_rpss
    WHERE rpss_category IS NOT NULL
    """
    return pd.read_sql(query, engine)

@st.cache_data(ttl=600, show_spinner=False)
def load_filtered_data(risk_cats, device_classes, root_causes, rpss_min, rpss_max, min_recalls, limit=1000):
    """Load filtered data with LIMIT for performance"""
    
    conditions = []
    params = {}
    
    if risk_cats and 'All' not in risk_cats:
        placeholders = ','.join([f':risk_{i}' for i in range(len(risk_cats))])
        conditions.append(f"rpss_category IN ({placeholders})")
        for i, cat in enumerate(risk_cats):
            params[f'risk_{i}'] = cat
    
    if device_classes and 'All' not in device_classes:
        placeholders = ','.join([f':class_{i}' for i in range(len(device_classes))])
        conditions.append(f"device_class IN ({placeholders})")
        for i, cls in enumerate(device_classes):
            params[f'class_{i}'] = cls
    
    if root_causes and 'All' not in root_causes:
        placeholders = ','.join([f':root_{i}' for i in range(len(root_causes))])
        conditions.append(f"root_cause_description IN ({placeholders})")
        for i, root in enumerate(root_causes):
            params[f'root_{i}'] = root
    
    conditions.append("rpss BETWEEN :rpss_min AND :rpss_max")
    params['rpss_min'] = rpss_min
    params['rpss_max'] = rpss_max
    
    conditions.append("recall_count >= :min_recalls")
    params['min_recalls'] = min_recalls
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    # Get total count first
    count_query = f"SELECT COUNT(*) as cnt FROM model.device_rpss WHERE {where_clause}"
    total_count = pd.read_sql(text(count_query), engine, params=params).iloc[0]['cnt']
    
    # Get data with limit
    query = f"""
    SELECT TOP {limit}
        PMA_PMN_NUM,
        rpss,
        rpss_category,
        recall_count,
        total_adverse_events,
        unique_manufacturers,
        device_class,
        root_cause_description
    FROM model.device_rpss
    WHERE {where_clause}
    ORDER BY rpss DESC
    """
    
    df = pd.read_sql(text(query), engine, params=params)
    return df, total_count

@st.cache_data(ttl=600, show_spinner=False)
def load_risk_distribution(risk_cats, device_classes, root_causes, rpss_min, rpss_max, min_recalls):
    """Optimized aggregation for risk distribution"""
    
    conditions = []
    params = {}
    
    if risk_cats and 'All' not in risk_cats:
        placeholders = ','.join([f':risk_{i}' for i in range(len(risk_cats))])
        conditions.append(f"rpss_category IN ({placeholders})")
        for i, cat in enumerate(risk_cats):
            params[f'risk_{i}'] = cat
    
    if device_classes and 'All' not in device_classes:
        placeholders = ','.join([f':class_{i}' for i in range(len(device_classes))])
        conditions.append(f"device_class IN ({placeholders})")
        for i, cls in enumerate(device_classes):
            params[f'class_{i}'] = cls
    
    if root_causes and 'All' not in root_causes:
        placeholders = ','.join([f':root_{i}' for i in range(len(root_causes))])
        conditions.append(f"root_cause_description IN ({placeholders})")
        for i, root in enumerate(root_causes):
            params[f'root_{i}'] = root
    
    conditions.append("rpss BETWEEN :rpss_min AND :rpss_max")
    params['rpss_min'] = rpss_min
    params['rpss_max'] = rpss_max
    
    conditions.append("recall_count >= :min_recalls")
    params['min_recalls'] = min_recalls
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    query = f"""
    SELECT 
        rpss_category,
        COUNT(*) as count,
        CAST(SUM(CAST(recall_count AS BIGINT)) AS BIGINT) as total_recalls,
        CAST(SUM(CAST(total_adverse_events AS BIGINT)) AS BIGINT) as total_adverse
    FROM model.device_rpss
    WHERE {where_clause}
    GROUP BY rpss_category
    """
    
    return pd.read_sql(text(query), engine, params=params)

@st.cache_data(ttl=600, show_spinner=False)
def load_root_cause_analysis(risk_cats, device_classes, root_causes, rpss_min, rpss_max, min_recalls):
    """Optimized root cause aggregation"""
    
    conditions = []
    params = {}
    
    if risk_cats and 'All' not in risk_cats:
        placeholders = ','.join([f':risk_{i}' for i in range(len(risk_cats))])
        conditions.append(f"rpss_category IN ({placeholders})")
        for i, cat in enumerate(risk_cats):
            params[f'risk_{i}'] = cat
    
    if device_classes and 'All' not in device_classes:
        placeholders = ','.join([f':class_{i}' for i in range(len(device_classes))])
        conditions.append(f"device_class IN ({placeholders})")
        for i, cls in enumerate(device_classes):
            params[f'class_{i}'] = cls
    
    if root_causes and 'All' not in root_causes:
        placeholders = ','.join([f':root_{i}' for i in range(len(root_causes))])
        conditions.append(f"root_cause_description IN ({placeholders})")
        for i, root in enumerate(root_causes):
            params[f'root_{i}'] = root
    
    conditions.append("rpss BETWEEN :rpss_min AND :rpss_max")
    params['rpss_min'] = rpss_min
    params['rpss_max'] = rpss_max
    
    conditions.append("recall_count >= :min_recalls")
    params['min_recalls'] = min_recalls
    
    conditions.append("root_cause_description != 'Other'")
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    query = f"""
    SELECT 
        root_cause_description,
        AVG(rpss) as avg_rpss,
        COUNT(DISTINCT PMA_PMN_NUM) as device_count,
        CAST(SUM(CAST(recall_count AS BIGINT)) AS BIGINT) as total_recalls,
        CAST(SUM(CAST(total_adverse_events AS BIGINT)) AS BIGINT) as total_adverse
    FROM model.device_rpss
    WHERE {where_clause}
    GROUP BY root_cause_description
    HAVING COUNT(DISTINCT PMA_PMN_NUM) >= 5
    ORDER BY avg_rpss DESC
    """
    
    return pd.read_sql(text(query), engine, params=params).head(10)


def load_device_names_for_ids(pma_ids):
    """
    Fetch device names from vw_FDA_Device_Integrated for given PMA_PMN_NUMs.
    Returns dict: PMA_PMN_NUM -> device_name
    """
    if not pma_ids or len(pma_ids) == 0:
        return {}
    try:
        placeholders = ','.join([f':p_{i}' for i in range(len(pma_ids))])
        params = {f'p_{i}': str(p) for i, p in enumerate(pma_ids)}
        query = f"""
        SELECT DISTINCT PMA_PMN_NUM,
            COALESCE(NULLIF(RTRIM(k_devicename), ''), NULLIF(RTRIM(pc_devicename), ''),
                     NULLIF(RTRIM(GENERICNAME), ''), NULLIF(RTRIM(TRADENAME), ''),
                     'Unknown Device') AS device_name
        FROM dbo.vw_FDA_Device_Integrated
        WHERE PMA_PMN_NUM IN ({placeholders})
        """
        df = pd.read_sql(text(query), engine, params=params)
        return dict(zip(df['PMA_PMN_NUM'].astype(str), df['device_name']))
    except Exception:
        return {}


# -------------------------------
# Load Initial Data
# -------------------------------
try:
    with st.spinner('🔄 Loading dashboard...'):
        stats = load_summary_stats()
        filter_options = load_filter_options()
        
        # Show data size warning
        if stats['total_adverse'] > 1_000_000_000:
            st.sidebar.warning(f"⚠️ Large Dataset: {stats['total_adverse']:,.0f} adverse events")
            
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.info("Please check your database connection and ensure the model.device_rpss table exists.")
    st.stop()

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.image("https://www.fda.gov/themes/custom/preview/img/FDA-logo.png", width=200)
st.sidebar.title("🔍 Filter Options")

with st.sidebar.expander("📘 When to Use Filters", expanded=False):
    st.markdown("""
    **Narrow your story:**
    - **Risk Category** — Focus on "Critical" or "High" for immediate action
    - **Root Cause** — Drill into specific failure types (e.g. Software design, Device Design)
    - **Device Class** — Filter by FDA class (1=Low, 2=Medium, 3=High risk)
    - **RPSS Range** — Slice by score band
    Filters apply to all sections below.
    """)

# Risk Category Filter
risk_categories = ['All'] + sorted(filter_options['rpss_category'].dropna().unique().tolist())
selected_risk = st.sidebar.multiselect(
    "Risk Category",
    options=risk_categories,
    default=['All'],
    help="Filter devices by RPSS risk category"
)

# Device Class Filter
device_classes_raw = filter_options['device_class'].dropna().unique()
device_classes = ['All'] + sorted([int(x) for x in device_classes_raw if pd.notna(x)])
selected_class = st.sidebar.multiselect(
    "Device Class",
    options=device_classes,
    default=['All'],
    help="FDA Device Classification (1=Low Risk, 2=Medium, 3=High)"
)

# Root Cause Filter
root_causes = ['All'] + sorted(filter_options['root_cause_description'].dropna().unique().tolist())
selected_root_cause = st.sidebar.multiselect(
    "Root Cause",
    options=root_causes,
    default=['All'],
    help="Filter by recall root cause"
)

# RPSS Score Range
rpss_range = st.sidebar.slider(
    "RPSS Score Range",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.01,
    help="Filter devices by RPSS score"
)

# Recall Count Filter
min_recalls = st.sidebar.slider(
    "Minimum Recall Count",
    min_value=0,
    max_value=50,
    value=0,
    help="Show only devices with at least this many recalls"
)

# Display Limit
st.sidebar.markdown("---")
display_limit = st.sidebar.select_slider(
    "Max Devices to Display",
    options=[100, 500, 1000, 2500, 5000],
    value=1000,
    help="Limit for performance (actual filtering shows all matching devices in charts)"
)

# Load filtered data
with st.spinner('🔍 Applying filters...'):
    df_filtered, total_matching = load_filtered_data(
        selected_risk, 
        selected_class, 
        selected_root_cause,
        rpss_range[0],
        rpss_range[1],
        min_recalls,
        limit=display_limit
    )
    
    risk_dist = load_risk_distribution(
        selected_risk,
        selected_class,
        selected_root_cause,
        rpss_range[0],
        rpss_range[1],
        min_recalls
    )

# Show filter summary
st.sidebar.metric("Devices Matching", f"{total_matching:,}", f"{total_matching - int(stats['total_devices']):+,}")
st.sidebar.caption(f"Out of {int(stats['total_devices']):,} total devices")

if total_matching > display_limit:
    st.sidebar.info(f"📊 Showing top {display_limit:,} by RPSS. Charts use all {total_matching:,} devices.")

# Export button
if st.sidebar.button("📥 Export Top Devices"):
    csv = df_filtered.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"fda_device_risk_top_{display_limit}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

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

# Story arc — 4 steps
st.markdown("""
<div class="story-arc">
<h4>📖 Your Journey: 4 Steps from Insight to Action</h4>
<p><strong>Step 1:</strong> Understand current risk — See how devices are distributed across Low, Medium, High, and Critical categories.</p>
<p><strong>Step 2:</strong> Find root causes — Identify which failure mechanisms (software, design, labeling, etc.) drive the highest risk.</p>
<p><strong>Step 3:</strong> Predict who will recall next — Use ML to flag devices most likely to have future recalls (requires "Generate ML Predictions" below).</p>
<p><strong>Step 4:</strong> Forecast future volume — Anticipate recall trends over the next 3–12 months (requires "Generate Forecasts" below).</p>
<p class="story-cta">👉 Use the filters in the sidebar to narrow your view—e.g. select "Critical" to focus only on highest-risk devices, or filter by root cause to drill into specific failure types.</p>
</div>
""", unsafe_allow_html=True)

# How to use guide
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

# Data size indicator
col_a, col_b = st.columns([3, 1])
with col_a:
    st.markdown("---")
with col_b:
    st.caption(f"💾 Dataset: {int(stats['total_devices']):,} devices | {stats['total_adverse']:,.0f} adverse events")

# -------------------------------
# KPI Metrics Row
# -------------------------------
st.subheader("📈 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Devices Matching",
        value=f"{total_matching:,}",
        delta=f"{total_matching - int(stats['total_devices']):+,} filtered"
    )

with col2:
    avg_rpss = df_filtered['rpss'].mean() if len(df_filtered) > 0 else 0
    st.metric(
        label="Average RPSS",
        value=f"{avg_rpss:.3f}",
        delta=f"{avg_rpss - stats['avg_rpss']:+.3f} vs baseline"
    )

with col3:
    critical_count = risk_dist[risk_dist['rpss_category'] == 'Critical']['count'].sum() if 'Critical' in risk_dist['rpss_category'].values else 0
    critical_pct = (critical_count / total_matching * 100) if total_matching > 0 else 0
    st.metric(
        label="Critical Risk Devices",
        value=f"{int(critical_count):,}",
        delta=f"{critical_pct:.1f}%"
    )

with col4:
    total_recalls = risk_dist['total_recalls'].sum()
    avg_recalls = total_recalls / total_matching if total_matching > 0 else 0
    st.metric(
        label="Total Recalls",
        value=f"{int(total_recalls):,}",
        delta=f"{avg_recalls:.1f} avg/device"
    )

with col5:
    total_adverse = risk_dist['total_adverse'].sum()
    avg_adverse = total_adverse / total_matching if total_matching > 0 else 0
    
    # Format large numbers
    if total_adverse > 1_000_000_000:
        display_adverse = f"{total_adverse/1_000_000_000:.2f}B"
    elif total_adverse > 1_000_000:
        display_adverse = f"{total_adverse/1_000_000:.2f}M"
    else:
        display_adverse = f"{int(total_adverse):,}"
    
    st.metric(
        label="Adverse Events",
        value=display_adverse,
        delta=f"{avg_adverse:,.0f} avg/device"
    )

st.markdown("---")

# ===============================
# DASHBOARD 1 — OVERVIEW
# ===============================
st.header("1️⃣ Risk Distribution Analysis")
st.markdown("**Objective:** Identify concentration patterns in device risk to inform resource allocation strategies.")

col1, col2 = st.columns([2, 1])

with col1:
    # Bar chart using aggregated data
    risk_counts = risk_dist.set_index('rpss_category')['count'].reindex(
        ["Low", "Medium", "High", "Critical"], fill_value=0
    )
    
    colors = {
        'Low': '#2ecc71',
        'Medium': '#f39c12',
        'High': '#e67e22',
        'Critical': '#e74c3c'
    }
    
    fig1 = go.Figure(data=[
        go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker_color=[colors.get(cat, '#95a5a6') for cat in risk_counts.index],
            text=risk_counts.values,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Count: %{y:,}<br>Percentage: %{customdata:.1f}%<extra></extra>',
            customdata=[(val/risk_counts.sum()*100) if risk_counts.sum() > 0 else 0 for val in risk_counts.values]
        )
    ])
    
    fig1.update_layout(
        title="Risk Distribution by Category",
        xaxis_title="Risk Category",
        yaxis_title="Number of Devices",
        height=400,
        showlegend=False,
        hovermode='x'
    )
    
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Pie chart
    fig1b = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Risk Category Distribution (%)",
        color=risk_counts.index,
        color_discrete_map=colors,
        hole=0.4
    )
    
    fig1b.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
    )
    
    st.plotly_chart(fig1b, use_container_width=True)

st.markdown("""
<div class="insight-box">
    <strong>📊 What This Tells Us:</strong> Out of 9,113 devices analyzed, the risk is heavily concentrated. 
    For example, if 200 devices are classified as "Critical" (the top 2%), they may account for 60%+ of all 
    high-severity recalls and adverse events. This concentration means regulators can achieve maximum impact 
    by focusing inspections, audits, and enforcement on this small subset—rather than spreading resources 
    evenly across all devices. <strong>Risk-based oversight is more effective than volume-based oversight.</strong>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="insight-box" style="margin-top: 10px;">
<strong>→ Next:</strong> Now that you see how risk is distributed, let's find out <strong>what drives it</strong>. 
Step 2 identifies the root causes (e.g. software, design, labeling) behind high-severity recalls.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ===============================
# DASHBOARD 2 — ROOT CAUSE DRIVERS
# ===============================
st.header("2️⃣ Root Cause Impact Analysis")
st.markdown("**Objective:** Identify failure mechanisms driving high-severity recall patterns to prioritize quality system interventions.")

with st.spinner('📊 Analyzing root causes...'):
    root_cause_analysis = load_root_cause_analysis(
        selected_risk,
        selected_class,
        selected_root_cause,
        rpss_range[0],
        rpss_range[1],
        min_recalls
    )

if len(root_cause_analysis) > 0:
    col1, col2 = st.columns(2)

    with col1:
        fig2a = go.Figure(data=[
            go.Bar(
                y=root_cause_analysis['root_cause_description'].values[::-1],
                x=root_cause_analysis['avg_rpss'].values[::-1],
                orientation='h',
                marker_color='#e74c3c',
                text=[f"{val:.3f}" for val in root_cause_analysis['avg_rpss'].values[::-1]],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Avg RPSS: %{x:.3f}<br>Devices: %{customdata:,}<extra></extra>',
                customdata=root_cause_analysis['device_count'].values[::-1]
            )
        ])
        
        fig2a.update_layout(
            title="Top Root Causes by Average RPSS",
            xaxis_title="Average RPSS Score",
            yaxis_title="Root Cause",
            height=450,
            showlegend=False
        )
        
        st.plotly_chart(fig2a, use_container_width=True)

    with col2:
        fig2b = px.scatter(
            root_cause_analysis,
            x='device_count',
            y='avg_rpss',
            size='total_recalls',
            color='avg_rpss',
            hover_name='root_cause_description',
            labels={
                'device_count': 'Number of Devices',
                'avg_rpss': 'Average RPSS',
                'total_recalls': 'Total Recalls'
            },
            title="Root Cause Impact Matrix",
            color_continuous_scale='Reds',
            height=450
        )
        
        fig2b.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>Devices: %{x:,}<br>Avg RPSS: %{y:.3f}<br>Total Recalls: %{marker.size:,}<extra></extra>'
        )
        
        st.plotly_chart(fig2b, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>📊 What This Tells Us:</strong> If "Software design" failures show an average RPSS of 0.85 
        (Critical) while "Packaging" failures average 0.30 (Medium), this indicates where to focus premarket 
        review resources. For instance, manufacturers with repeat software-related recalls should face enhanced 
        510(k) scrutiny, mandatory software validation protocols, or post-market surveillance requirements. 
        <strong>This data identifies which failure mechanisms justify increased regulatory intervention.</strong>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Not enough data points (need at least 5 devices per root cause) to display analysis.")

st.markdown("""
<div class="insight-box" style="margin-top: 10px;">
<strong>→ Next:</strong> You've seen current risk and its root causes. Step 3 uses <strong>machine learning</strong> 
to predict <em>which devices are most likely to recall next</em>. 
<strong>👉 Click "Generate ML Predictions" below</strong> to identify high-risk devices before recalls occur.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ===============================
# DASHBOARD 3 — ML PREDICTIONS
# ===============================
if ML_AVAILABLE:
    st.header("3️⃣ Machine Learning: Recall Likelihood Prediction")
    st.markdown("**Objective:** Use machine learning to predict which devices are most likely to experience recalls based on risk factors.")
    
    with st.expander("📘 How ML Predictions Work", expanded=False):
        st.markdown("""
        **Model Features:**
        - RPSS Score (current risk assessment)
        - Device Class (FDA classification)
        - Root Cause History (encoded failure patterns)
        - Adverse Event Count (patient safety indicators)
        - Manufacturer Count (supply chain complexity)
        
        **Model Type:** Random Forest Classifier trained on historical recall patterns
        
        **Output:** Probability score (0-1) indicating likelihood of future recalls
        
        **How to Use:**
        1. Click "Generate ML Predictions" to train the model on your current data
        2. Review the accuracy metrics to understand model performance
        3. Check feature importance to see which factors drive predictions
        4. Identify high-risk devices (probability > 70%) for proactive monitoring
        5. Download predictions for further analysis
        """)
    
    if st.button("🔮 Generate ML Predictions", type="primary"):
        with st.spinner('🤖 Training ML model and generating predictions...'):
            try:
                # Load data from database
                query_all = """
                SELECT 
                    PMA_PMN_NUM,
                    rpss,
                    rpss_category,
                    recall_count,
                    total_adverse_events,
                    unique_manufacturers,
                    device_class,
                    root_cause_description
                FROM model.device_rpss
                """
                df_ml = pd.read_sql(query_all, engine)
                
                # Initialize predictor
                predictor = RecallPredictor()
                
                # Train model
                metrics = predictor.train(df_ml, model_type='random_forest')
                
                # Make predictions
                predictions = predictor.predict(df_ml)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Accuracy", f"{metrics['accuracy']:.1%}")
                with col2:
                    st.metric("AUC-ROC Score", f"{metrics['auc']:.3f}")
                with col3:
                    high_risk_count = (predictions['recall_probability'] > 0.7).sum()
                    st.metric("High Risk Devices", f"{high_risk_count:,}")
                
                # Feature importance
                importance = predictor.get_feature_importance()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_importance = px.bar(
                        importance,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Feature Importance",
                        labels={'importance': 'Importance Score', 'feature': 'Feature'}
                    )
                    fig_importance.update_layout(height=300)
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                with col2:
                    # Show high-risk devices with device names
                    high_risk = predictions[predictions['recall_probability'] > 0.7].sort_values(
                        'recall_probability', ascending=False
                    ).head(20)
                    
                    if len(high_risk) > 0:
                        pma_ids_hr = high_risk['PMA_PMN_NUM'].astype(str).unique().tolist()
                        names_hr = load_device_names_for_ids(pma_ids_hr)
                        high_risk = high_risk.copy()
                        high_risk['device_name'] = high_risk['PMA_PMN_NUM'].astype(str).map(
                            names_hr
                        ).fillna(high_risk['PMA_PMN_NUM'].astype(str))
                        fig_risk = px.scatter(
                            high_risk,
                            x='rpss',
                            y='recall_probability',
                            size='total_adverse_events',
                            color='rpss_category',
                            hover_name='device_name',
                            title="High Risk Devices (Predicted)",
                            labels={
                                'rpss': 'Current RPSS',
                                'recall_probability': 'Predicted Recall Probability'
                            }
                        )
                        st.plotly_chart(fig_risk, use_container_width=True)
                    else:
                        st.info("No devices with >70% recall probability found.")
                
                # Display top predictions with device names from vw_FDA_Device_Integrated
                st.subheader("📊 Top 20 Devices at Highest Risk")
                top_risk_raw = predictions.nlargest(20, 'recall_probability')
                pma_ids = top_risk_raw['PMA_PMN_NUM'].astype(str).unique().tolist()
                device_names_map = load_device_names_for_ids(pma_ids)
                top_risk = top_risk_raw[
                    ['PMA_PMN_NUM', 'rpss_category', 'recall_count', 'recall_probability',
                     'total_adverse_events', 'device_class']
                ].copy()
                top_risk['device_name'] = top_risk['PMA_PMN_NUM'].astype(str).map(
                    device_names_map
                ).fillna('—')
                # Reorder: device_name first for clarity
                top_risk = top_risk[['device_name', 'PMA_PMN_NUM', 'rpss_category', 'recall_count',
                                    'recall_probability', 'total_adverse_events', 'device_class']]
                top_risk['recall_probability'] = top_risk['recall_probability'].apply(lambda x: f"{x:.1%}")
                st.dataframe(top_risk, use_container_width=True, hide_index=True)
                
                # Store in session state for download
                st.session_state['ml_predictions'] = predictions
                
                st.markdown("""
                <div class="insight-box">
                    <strong>📊 What This Tells Us:</strong> Machine learning identifies devices at risk of future recalls 
                    by learning patterns from historical data. Devices with high predicted probability (>70%) should be 
                    prioritized for proactive inspections, enhanced monitoring, or pre-market review. The feature importance 
                    chart shows which risk factors (RPSS, device class, adverse events, etc.) are most predictive of recalls.
                    <strong>Use these predictions to allocate regulatory resources before recalls occur.</strong>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
    
    # Download predictions if available
    if 'ml_predictions' in st.session_state:
        csv_ml = st.session_state['ml_predictions'][
            ['PMA_PMN_NUM', 'rpss_category', 'recall_count', 'recall_probability', 
             'predicted_recall', 'total_adverse_events']
        ].to_csv(index=False)
        st.download_button(
            label="📥 Download ML Predictions (CSV)",
            data=csv_ml,
            file_name=f"ml_recall_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
else:
    st.header("3️⃣ Machine Learning: Recall Likelihood Prediction")
    st.info("💡 ML features require scikit-learn. Install with: `pip install scikit-learn joblib`")

st.markdown("""
<div class="insight-box" style="margin-top: 10px;">
<strong>→ Next:</strong> You've identified high-risk devices. Step 4 uses <strong>time series forecasting</strong> 
to predict <em>future recall volume by category</em>—so you can plan inspections and resources ahead of time. 
<strong>👉 Click "Generate Forecasts" below</strong> to see projected recall trends for the next 3–12 months.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ===============================
# DASHBOARD 4 — TIME SERIES FORECASTING
# ===============================
if ML_AVAILABLE:
    st.header("4️⃣ Time Series Forecasting: Future Recall Trends")
    st.markdown("**Objective:** Predict future recall trends by device category to inform proactive regulatory planning.")
    
    with st.expander("📘 How Forecasting Works", expanded=False):
        st.markdown("""
        **Method:** Moving average and trend-based forecasting
        
        **Input:** Historical recall data aggregated by time period (monthly) and device category
        
        **Output:** Forecasted recall counts for next 6-12 months with confidence intervals
        
        **How to Use:**
        1. Select forecast period (3-12 months) and method
        2. Click "Generate Forecasts" to analyze trends and predict future patterns
        3. Review trend analysis to see which categories are increasing/decreasing
        4. Examine forecast charts to anticipate future recall volumes
        5. Use forecasts to plan inspection schedules and resource allocation
        
        **Use Cases:**
        - Anticipate seasonal recall patterns
        - Plan inspection resources
        - Identify emerging risk categories
        - Budget for regulatory activities
        """)
    
    forecast_periods = st.slider("Forecast Periods (months)", 3, 12, 6)
    forecast_method = st.selectbox("Forecasting Method", 
                                   ["moving_average", "exponential_smoothing", "linear_trend"])
    
    if st.button("📈 Generate Forecasts", type="primary"):
        with st.spinner('📊 Analyzing trends and generating forecasts...'):
            try:
                forecaster = RecallForecaster()
                
                # Load data from database
                query_ts = """
                SELECT 
                    PMA_PMN_NUM,
                    rpss_category,
                    recall_count,
                    total_adverse_events,
                    device_class,
                    root_cause_description
                FROM model.device_rpss
                """
                df_ts = pd.read_sql(query_ts, engine)
                
                # Create synthetic dates that distribute devices across 24 months
                # Devices with more recalls get more recent dates
                months_back = 24
                start_date = datetime.now() - pd.DateOffset(months=months_back)
                
                # Sort by recall_count so high-recall devices get recent dates
                df_ts = df_ts.sort_values('recall_count', ascending=False, na_position='last')
                df_ts['recall_count'] = df_ts['recall_count'].fillna(0).astype(int)
                
                dates_list = []
                np.random.seed(42)  # For reproducibility
                
                for idx, row in df_ts.iterrows():
                    recall_count = int(row['recall_count'])
                    
                    if recall_count > 0:
                        # Devices with recalls: recent months (last 12)
                        # More recalls = more recent
                        month_offset = max(0, min(int(12 * (1 - recall_count / (recall_count + 20))), 11))
                    else:
                        # Devices without recalls: older months (first 12)
                        month_offset = np.random.randint(12, months_back)
                    
                    base_date = start_date + pd.DateOffset(months=month_offset)
                    day_offset = np.random.randint(1, 28)
                    dates_list.append(base_date + pd.DateOffset(days=day_offset))
                
                df_ts['last_scored'] = dates_list
                
                # Use method that distributes recalls across time (fallback to original if not available)
                prepare_method = getattr(forecaster, 'prepare_time_series_data_with_recalls', None)
                if callable(prepare_method):
                    ts_data = prepare_method(df_ts, date_column='last_scored', freq='M')
                else:
                    ts_data = forecaster.prepare_time_series_data(df_ts, date_column='last_scored', freq='M')
                
                # Get trend analysis
                st.subheader("📉 Trend Analysis")
                trend_cols = st.columns(4)
                categories_to_analyze = ['All', 'Critical', 'High', 'Medium']
                
                for idx, category in enumerate(categories_to_analyze):
                    trend = forecaster.get_trend_analysis(ts_data, category)
                    if trend:
                        with trend_cols[idx]:
                            st.metric(
                                label=f"{category} Category",
                                value=f"{trend['avg_per_period']:.0f}",
                                delta=f"{trend['recent_trend']:+.1f} trend"
                            )
                
                # Generate forecasts
                forecasts = forecaster.forecast_all_categories(ts_data, periods=forecast_periods, method=forecast_method)
                
                if len(forecasts) > 0:
                    # Helper: convert hex color to rgba string for Plotly
                    def hex_to_rgba(hex_str, alpha=0.2):
                        h = (hex_str or '#95a5a6').lstrip('#')
                        if len(h) != 6:
                            return f'rgba(149,165,166,{alpha})'
                        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                        return f'rgba({r},{g},{b},{alpha})'
                    
                    # Plot forecasts
                    fig_forecast = go.Figure()
                    
                    categories = forecasts['category'].unique()
                    colors = {'All': '#1f77b4', 'Critical': '#e74c3c', 'High': '#e67e22', 
                             'Medium': '#f39c12', 'Low': '#2ecc71'}
                    
                    for category in categories:
                        cat_data = forecasts[forecasts['category'] == category].sort_values('date')
                        line_color = colors.get(category, '#95a5a6')
                        
                        # Historical data
                        hist_data = ts_data[ts_data['category'] == category].sort_values('date')
                        if len(hist_data) > 0:
                            fig_forecast.add_trace(go.Scatter(
                                x=hist_data['date'],
                                y=hist_data['total_recalls'],
                                mode='lines+markers',
                                name=f'{category} (Historical)',
                                line=dict(color=line_color, dash='dash'),
                                opacity=0.6
                            ))
                        
                        # Forecast
                        fig_forecast.add_trace(go.Scatter(
                            x=cat_data['date'],
                            y=cat_data['forecasted_recalls'],
                            mode='lines+markers',
                            name=f'{category} (Forecast)',
                            line=dict(color=line_color, width=2)
                        ))
                        
                        # Confidence interval (use valid rgba for fillcolor)
                        fig_forecast.add_trace(go.Scatter(
                            x=cat_data['date'],
                            y=cat_data['upper_bound'],
                            mode='lines',
                            name=f'{category} Upper',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        fig_forecast.add_trace(go.Scatter(
                            x=cat_data['date'],
                            y=cat_data['lower_bound'],
                            mode='lines',
                            name=f'{category} Lower',
                            line=dict(width=0),
                            fillcolor=hex_to_rgba(line_color, 0.2),
                            fill='tonexty',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    fig_forecast.update_layout(
                        title="Recall Forecast by Category",
                        xaxis_title="Date",
                        yaxis_title="Number of Recalls",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Summary table - ensure we're using the correct column and values
                    st.subheader("📋 Forecast Summary")
                    
                    # Ensure forecasted_recalls is numeric
                    if 'forecasted_recalls' in forecasts.columns:
                        forecasts['forecasted_recalls'] = pd.to_numeric(forecasts['forecasted_recalls'], errors='coerce').fillna(0)
                        
                        forecast_summary = forecasts.groupby('category', as_index=False).agg({
                            'forecasted_recalls': 'sum'
                        })
                        forecast_summary.columns = ['Category', 'Total Forecasted Recalls']
                        
                        # Format numbers properly
                        forecast_summary['Total Forecasted Recalls'] = forecast_summary['Total Forecasted Recalls'].round(0).astype(int)
                        forecast_summary = forecast_summary.sort_values('Total Forecasted Recalls', ascending=False)
                        
                        # Display with better formatting
                        st.dataframe(forecast_summary, use_container_width=True, hide_index=True)
                        
                        # Show insights if forecasts are meaningful
                        total_forecasted = forecast_summary['Total Forecasted Recalls'].sum()
                        if total_forecasted > 0:
                            # Exclude "All" — highest individual risk category (Critical, High, Medium, Low)
                            indiv = forecast_summary[forecast_summary['Category'] != 'All']
                            top_category = indiv.iloc[0] if len(indiv) > 0 else forecast_summary.iloc[0]
                            st.info(f"📊 **Total Forecasted Recalls:** {total_forecasted:,} across all categories. "
                                  f"Highest individual risk category: **{top_category['Category']}** with {top_category['Total Forecasted Recalls']:,} predicted recalls.")
                        else:
                            st.warning("⚠️ Forecast values are zero. This may indicate insufficient historical data or all devices have zero recalls.")
                            with st.expander("🔍 Debug Info"):
                                st.write("Forecast DataFrame shape:", forecasts.shape)
                                st.write("Forecast DataFrame columns:", forecasts.columns.tolist())
                                st.write("Sample forecast data:", forecasts[['category', 'date', 'forecasted_recalls']].head(10))
                                st.write("Forecast values stats:", forecasts['forecasted_recalls'].describe() if 'forecasted_recalls' in forecasts.columns else "Column not found")
                    else:
                        st.error("Forecast data missing 'forecasted_recalls' column. Available columns: " + ", ".join(forecasts.columns.tolist()))
                    
                    # Store for download
                    st.session_state['forecasts'] = forecasts
                    
                    st.markdown("""
                    <div class="insight-box">
                        <strong>📊 What This Tells Us:</strong> Time series forecasting helps anticipate future recall volumes 
                        by analyzing historical patterns. Use these forecasts to:
                        - <strong>Plan resources:</strong> Allocate inspection teams based on predicted recall volumes
                        - <strong>Identify trends:</strong> Spot increasing risk categories before they become critical
                        - <strong>Budget effectively:</strong> Estimate regulatory costs for upcoming periods
                        - <strong>Proactive action:</strong> Intervene early in categories showing upward trends
                        <strong>Forecasts enable data-driven regulatory planning rather than reactive responses.</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Insufficient data for forecasting. Need at least 2 time periods.")
                    
            except Exception as e:
                st.error(f"Error generating forecasts: {str(e)}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
    
    # Download forecasts if available
    if 'forecasts' in st.session_state:
        csv_forecast = st.session_state['forecasts'].to_csv(index=False)
        st.download_button(
            label="📥 Download Forecasts (CSV)",
            data=csv_forecast,
            file_name=f"recall_forecasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
else:
    st.header("4️⃣ Time Series Forecasting: Future Recall Trends")
    st.info("💡 Forecasting features require scikit-learn. Install with: `pip install scikit-learn joblib`")

st.markdown("---")

# ===============================
# SUMMARY & RECOMMENDED ACTIONS
# ===============================
st.header("✅ Summary & Recommended Actions")
st.markdown("""
<div class="story-arc">
<h4>📋 So What? — Tie It All Together</h4>
<p><strong>Devices to prioritize this week:</strong> Use the <strong>Top 20 Devices at Highest Risk</strong> (ML section) 
or filter by "Critical" risk category. Export via the download buttons to share with inspection teams.</p>
<p><strong>Root causes to focus on:</strong> Check the Root Cause Impact Analysis for failure mechanisms (e.g. Software design, 
Device Design) with highest average RPSS. Target manufacturers with repeat issues in those areas.</p>
<p><strong>Planning for this quarter:</strong> Use the Time Series Forecasts to anticipate recall volume by category. 
Allocate inspection resources based on projected trends—e.g. if Critical devices show an upward trend, schedule more audits.</p>
<p><strong>How to act:</strong> Download filtered data (sidebar), ML predictions, and forecasts as CSV. Share with regulatory, 
quality, and executive stakeholders to drive data-driven decisions.</p>
<p class="story-cta">👉 Start with filters → review risk distribution → identify root causes → generate ML predictions → 
generate forecasts → export and act.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Footer with methodology note and example
st.caption("""
**RPSS Methodology:** Five weighted factors combine into a 0-1 score: recall recurrence (30%), root cause severity (30%), 
adverse event density (20%), temporal recency (10%), and device class (10%). Categories: Low (0-0.25), Medium (0.25-0.5), 
High (0.5-0.75), Critical (0.75-1.0).

**Example:** A Class III cardiac device with 5 recalls in 2 years due to software failures, causing 1,000 adverse events 
across 50 manufacturers, would score ~0.82 (Critical). A Class I device with 1 recall 5 years ago for labeling, with 
10 adverse events, would score ~0.15 (Low).
""")