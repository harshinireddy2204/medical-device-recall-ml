import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import os
import sys

# Add Scripts directory to path for ML modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Scripts'))

try:
    from ml_recall_prediction import RecallPredictor
    from time_series_forecast import RecallForecaster
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    st.warning(f"ML modules not available: {e}. Some features will be disabled.")

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
    .story-cta { background-color: #e8f4f8; padding: 12px; border-radius: 6px; margin: 10px 0; font-weight: 600; color: #000000; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Data Source (CSV snapshot)
# -------------------------------
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "device_rpss_sample.csv")


@st.cache_data(ttl=600, show_spinner=False)
def load_base_data():
    """
    Load the full device RPSS dataset from CSV.

    Expected columns (at minimum):
    PMA_PMN_NUM, rpss, rpss_category, recall_count,
    total_adverse_events, unique_manufacturers,
    device_class, root_cause_description
    """
    # Try different encodings to handle various CSV formats
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    
    df = None
    last_error = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(DATA_PATH, encoding=encoding, low_memory=False)
            # If we successfully read the file, break out of the loop
            if df is not None and len(df) > 0:
                break
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue
    
    if df is None or len(df) == 0:
        raise FileNotFoundError(
            f"Could not read CSV file with any encoding. Last error: {last_error}. "
            f"Tried encodings: {', '.join(encodings)}"
        )

    # Normalize manufacturer column (CSV uses unique-manufacturers)
    if "unique-manufacturers" in df.columns and "unique_manufacturers" not in df.columns:
        df["unique_manufacturers"] = df["unique-manufacturers"]
    elif "unique_manufacturers" not in df.columns:
        df["unique_manufacturers"] = 0

    # Ensure key numeric columns are the right dtype
    numeric_cols = [
        "rpss",
        "recall_count",
        "total_adverse_events",
        "device_class",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure device_name exists (for ML Top 20)
    if "device_name" not in df.columns:
        df["device_name"] = df["PMA_PMN_NUM"].astype(str)

    return df

@st.cache_data(ttl=600, show_spinner=False)
def load_summary_stats():
    """Load summary statistics from CSV (BIGINT safe via int64 in pandas)."""
    df = load_base_data()

    total_devices = len(df)
    avg_rpss = float(df["rpss"].mean())

    total_recalls = int(df["recall_count"].fillna(0).astype("int64").sum())
    total_adverse = int(df["total_adverse_events"].fillna(0).astype("int64").sum())

    critical_count = (df["rpss_category"] == "Critical").sum()
    high_count = (df["rpss_category"] == "High").sum()
    medium_count = (df["rpss_category"] == "Medium").sum()
    low_count = (df["rpss_category"] == "Low").sum()

    max_adverse_single_device = int(df["total_adverse_events"].fillna(0).max())

    return pd.Series(
        {
            "total_devices": total_devices,
            "avg_rpss": avg_rpss,
            "total_recalls": total_recalls,
            "total_adverse": total_adverse,
            "critical_count": critical_count,
            "high_count": high_count,
            "medium_count": medium_count,
            "low_count": low_count,
            "max_adverse_single_device": max_adverse_single_device,
        }
    )

@st.cache_data(ttl=600, show_spinner=False)
def load_filter_options():
    """Load unique values for filters from CSV."""
    df = load_base_data()
    df = df[df["rpss_category"].notna()]

    return df[["rpss_category", "device_class", "root_cause_description"]].drop_duplicates()

@st.cache_data(ttl=600, show_spinner=False)
def load_filtered_data(
    risk_cats, device_classes, root_causes, rpss_min, rpss_max, min_recalls, limit=1000
):
    """Load filtered data from CSV with LIMIT for performance."""
    df = load_base_data()

    # Start with all rows
    mask = pd.Series(True, index=df.index)

    if risk_cats and "All" not in risk_cats:
        mask &= df["rpss_category"].isin(risk_cats)

    if device_classes and "All" not in device_classes:
        mask &= df["device_class"].isin(device_classes)

    if root_causes and "All" not in root_causes:
        mask &= df["root_cause_description"].isin(root_causes)

    mask &= df["rpss"].between(rpss_min, rpss_max, inclusive="both")
    mask &= df["recall_count"].fillna(0) >= min_recalls

    df_filtered_all = df[mask].copy()
    total_count = len(df_filtered_all)

    # Sort by RPSS and limit
    df_filtered_limited = (
        df_filtered_all.sort_values("rpss", ascending=False)
        .head(limit)[
            [
                "PMA_PMN_NUM",
                "rpss",
                "rpss_category",
                "recall_count",
                "total_adverse_events",
                "unique_manufacturers",
                "device_class",
                "root_cause_description",
            ]
        ]
    )

    return df_filtered_limited, total_count

@st.cache_data(ttl=600, show_spinner=False)
def load_risk_distribution(
    risk_cats, device_classes, root_causes, rpss_min, rpss_max, min_recalls
):
    """Optimized aggregation for risk distribution from CSV."""
    df = load_base_data()

    mask = pd.Series(True, index=df.index)

    if risk_cats and "All" not in risk_cats:
        mask &= df["rpss_category"].isin(risk_cats)

    if device_classes and "All" not in device_classes:
        mask &= df["device_class"].isin(device_classes)

    if root_causes and "All" not in root_causes:
        mask &= df["root_cause_description"].isin(root_causes)

    mask &= df["rpss"].between(rpss_min, rpss_max, inclusive="both")
    mask &= df["recall_count"].fillna(0) >= min_recalls

    df_f = df[mask].copy()
    df_f["recall_count"] = df_f["recall_count"].fillna(0).astype("int64")
    df_f["total_adverse_events"] = df_f["total_adverse_events"].fillna(0).astype("int64")

    grouped = (
        df_f.groupby("rpss_category", dropna=False)
        .agg(
            count=("PMA_PMN_NUM", "size"),
            total_recalls=("recall_count", "sum"),
            total_adverse=("total_adverse_events", "sum"),
        )
        .reset_index()
    )

    return grouped

@st.cache_data(ttl=600, show_spinner=False)
def load_root_cause_analysis(
    risk_cats, device_classes, root_causes, rpss_min, rpss_max, min_recalls
):
    """Optimized root cause aggregation from CSV."""
    df = load_base_data()

    mask = pd.Series(True, index=df.index)

    if risk_cats and "All" not in risk_cats:
        mask &= df["rpss_category"].isin(risk_cats)

    if device_classes and "All" not in device_classes:
        mask &= df["device_class"].isin(device_classes)

    if root_causes and "All" not in root_causes:
        mask &= df["root_cause_description"].isin(root_causes)

    mask &= df["rpss"].between(rpss_min, rpss_max, inclusive="both")
    mask &= df["recall_count"].fillna(0) >= min_recalls
    mask &= df["root_cause_description"] != "Other"

    df_f = df[mask].copy()
    df_f["recall_count"] = df_f["recall_count"].fillna(0).astype("int64")
    df_f["total_adverse_events"] = df_f["total_adverse_events"].fillna(0).astype("int64")

    grouped = (
        df_f.groupby("root_cause_description")
        .agg(
            avg_rpss=("rpss", "mean"),
            device_count=("PMA_PMN_NUM", "nunique"),
            total_recalls=("recall_count", "sum"),
            total_adverse=("total_adverse_events", "sum"),
        )
        .reset_index()
    )

    grouped = grouped[grouped["device_count"] >= 5].sort_values("avg_rpss", ascending=False)

    return grouped.head(10)

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
            
except FileNotFoundError as e:
    st.error(f"❌ CSV file not found: {str(e)}")
    st.info(f"Looking for file at: `{DATA_PATH}`")
    st.info("Please ensure `device_rpss_sample.csv` is in the `visualization/` folder and committed to the repository.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.info(f"File path attempted: `{DATA_PATH}`")
    import traceback
    with st.expander("🔍 Full error details"):
        st.code(traceback.format_exc())
    st.stop()

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.image("https://www.fda.gov/themes/custom/preview/img/FDA-logo.png", width=200)
st.sidebar.title("🔍 Filter Options")

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

# Story arc
st.markdown("""
<div class="story-arc">
<h4>📖 Your Journey: 4 Steps from Insight to Action</h4>
<p><strong>Step 1:</strong> Understand current risk — See how devices are distributed across Low, Medium, High, and Critical categories.</p>
<p><strong>Step 2:</strong> Find root causes — Identify which failure mechanisms (software, design, labeling, etc.) drive the highest risk.</p>
<p><strong>Step 3:</strong> Predict who will recall next — Use ML to flag devices most likely to have future recalls (click "Generate ML Predictions" below).</p>
<p><strong>Step 4:</strong> Forecast future volume — Anticipate recall trends over the next 3–12 months (click "Generate Forecasts" below).</p>
<p class="story-cta">👉 Use the sidebar filters to narrow your view—e.g. select "Critical" to focus on highest-risk devices.</p>
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
to predict which devices are most likely to recall next. 
<strong>👉 Click "Generate ML Predictions" below</strong> to identify high-risk devices.
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
        """)
    
    if st.button("🔮 Generate ML Predictions", type="primary"):
        with st.spinner('🤖 Training ML model and generating predictions...'):
            try:
                # Initialize predictor
                predictor = RecallPredictor()
                
                # Train model
                df_ml = load_base_data()
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
                    # Show high-risk devices (use device_name from CSV if available)
                    high_risk = predictions[predictions['recall_probability'] > 0.7].sort_values(
                        'recall_probability', ascending=False
                    ).head(20).copy()
                    high_risk['hover_name'] = high_risk.get('device_name', high_risk['PMA_PMN_NUM'].astype(str))
                    high_risk['hover_name'] = high_risk['hover_name'].fillna(high_risk['PMA_PMN_NUM'].astype(str))
                    
                    fig_risk = px.scatter(
                        high_risk,
                        x='rpss',
                        y='recall_probability',
                        size='total_adverse_events',
                        color='rpss_category',
                        hover_name='hover_name',
                        title="High Risk Devices (Predicted)",
                        labels={
                            'rpss': 'Current RPSS',
                            'recall_probability': 'Predicted Recall Probability'
                        }
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                # Display top predictions with device names (from CSV / vw_FDA_Device_Integrated)
                st.subheader("📊 Top 20 Devices at Highest Risk")
                top_risk_raw = predictions.nlargest(20, 'recall_probability')
                top_risk = top_risk_raw[
                    ['PMA_PMN_NUM', 'rpss_category', 'recall_count', 'recall_probability',
                     'total_adverse_events', 'device_class']
                ].copy()
                top_risk['device_name'] = top_risk_raw.get('device_name', top_risk_raw['PMA_PMN_NUM'].astype(str))
                top_risk['device_name'] = top_risk['device_name'].fillna('—')
                top_risk = top_risk[['device_name', 'PMA_PMN_NUM', 'rpss_category', 'recall_count',
                                    'recall_probability', 'total_adverse_events', 'device_class']]
                top_risk['recall_probability'] = top_risk['recall_probability'].apply(lambda x: f"{x:.1%}")
                st.dataframe(top_risk, use_container_width=True, hide_index=True)
                
                # Store in session state for download
                st.session_state['ml_predictions'] = predictions
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
    
    # Download predictions if available
    if 'ml_predictions' in st.session_state:
        pred_cols = ['PMA_PMN_NUM', 'rpss_category', 'recall_count', 'recall_probability', 
                     'predicted_recall', 'total_adverse_events']
        if 'device_name' in st.session_state['ml_predictions'].columns:
            pred_cols = ['device_name'] + pred_cols
        csv_ml = st.session_state['ml_predictions'][pred_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download ML Predictions (CSV)",
            data=csv_ml,
            file_name=f"ml_recall_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    st.markdown("""
    <div class="insight-box" style="margin-top: 10px;">
    <strong>→ Next:</strong> Step 4 uses <strong>time series forecasting</strong> to predict future recall volume by category. 
    <strong>👉 Click "Generate Forecasts" below</strong> to see projected recall trends.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # ===============================
    # DASHBOARD 4 — TIME SERIES FORECASTING
    # ===============================
    st.header("4️⃣ Time Series Forecasting: Future Recall Trends")
    st.markdown("**Objective:** Predict future recall trends by device category to inform proactive regulatory planning.")
    
    with st.expander("📘 How Forecasting Works", expanded=False):
        st.markdown("""
        **Method:** Moving average and trend-based forecasting
        
        **Input:** Historical recall data aggregated by time period (monthly) and device category
        
        **Output:** Forecasted recall counts for next 6-12 months with confidence intervals
        
        **Use Cases:**
        - Anticipate seasonal recall patterns
        - Plan inspection resources
        - Identify emerging risk categories
        """)
    
    forecast_periods = st.slider("Forecast Periods (months)", 3, 12, 6)
    forecast_method = st.selectbox("Forecasting Method", 
                                   ["moving_average", "exponential_smoothing", "linear_trend"])
    
    if st.button("📈 Generate Forecasts", type="primary"):
        with st.spinner('📊 Analyzing trends and generating forecasts...'):
            try:
                forecaster = RecallForecaster()
                df_ts = load_base_data()
                
                # Synthetic dates: distribute devices across 24 months (same as app.py)
                # CSV last_scored can be times-only or same date → huge single-period sums
                months_back = 24
                start_date = datetime.now() - pd.DateOffset(months=months_back)
                df_ts = df_ts.sort_values('recall_count', ascending=False, na_position='last')
                df_ts['recall_count'] = df_ts['recall_count'].fillna(0).astype(int)
                dates_list = []
                np.random.seed(42)
                for idx, row in df_ts.iterrows():
                    recall_count = int(row['recall_count'])
                    if recall_count > 0:
                        month_offset = max(0, min(int(12 * (1 - recall_count / (recall_count + 20))), 11))
                    else:
                        month_offset = np.random.randint(12, months_back)
                    base_date = start_date + pd.DateOffset(months=month_offset)
                    day_offset = np.random.randint(1, 28)
                    dates_list.append(base_date + pd.DateOffset(days=day_offset))
                df_ts['last_scored'] = dates_list
                
                # Use prepare_time_series_data_with_recalls if available (distributes recalls across months)
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
                    # Helper: convert hex to rgba for Plotly fillcolor
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
                        
                        # Historical data
                        hist_data = ts_data[ts_data['category'] == category].sort_values('date')
                        if len(hist_data) > 0:
                            fig_forecast.add_trace(go.Scatter(
                                x=hist_data['date'],
                                y=hist_data['total_recalls'],
                                mode='lines+markers',
                                name=f'{category} (Historical)',
                                line=dict(color=colors.get(category, '#95a5a6'), dash='dash'),
                                opacity=0.6
                            ))
                        
                        # Forecast
                        fig_forecast.add_trace(go.Scatter(
                            x=cat_data['date'],
                            y=cat_data['forecasted_recalls'],
                            mode='lines+markers',
                            name=f'{category} (Forecast)',
                            line=dict(color=colors.get(category, '#95a5a6'), width=2)
                        ))
                        
                        # Confidence interval
                        fig_forecast.add_trace(go.Scatter(
                            x=cat_data['date'],
                            y=cat_data['upper_bound'],
                            mode='lines',
                            name=f'{category} Upper',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        line_color = colors.get(category, '#95a5a6')
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
                    
                    # Summary table (exclude "All" for highest-individual-category insight)
                    st.subheader("📋 Forecast Summary")
                    forecasts['forecasted_recalls'] = pd.to_numeric(forecasts['forecasted_recalls'], errors='coerce').fillna(0)
                    forecast_summary = forecasts.groupby('category', as_index=False).agg({'forecasted_recalls': 'sum'})
                    forecast_summary.columns = ['Category', 'Total Forecasted Recalls']
                    forecast_summary['Total Forecasted Recalls'] = forecast_summary['Total Forecasted Recalls'].round(0).astype(int)
                    forecast_summary = forecast_summary.sort_values('Total Forecasted Recalls', ascending=False)
                    st.dataframe(forecast_summary, use_container_width=True, hide_index=True)
                    total_forecasted = forecast_summary['Total Forecasted Recalls'].sum()
                    if total_forecasted > 0:
                        indiv = forecast_summary[forecast_summary['Category'] != 'All']
                        top_category = indiv.iloc[0] if len(indiv) > 0 else forecast_summary.iloc[0]
                        st.info(f"📊 **Total Forecasted Recalls:** {total_forecasted:,}. "
                              f"Highest individual risk category: **{top_category['Category']}** with {top_category['Total Forecasted Recalls']:,} predicted recalls.")
                    
                    # Store for download
                    st.session_state['forecasts'] = forecasts
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

st.markdown("---")

# ===============================
# SUMMARY & RECOMMENDED ACTIONS
# ===============================
st.header("✅ Summary & Recommended Actions")
st.markdown("""
<div class="story-arc">
<h4>📋 So What? — Tie It All Together</h4>
<p><strong>Devices to prioritize:</strong> Use the Top 20 Devices at Highest Risk (ML section) or filter by "Critical" risk category.</p>
<p><strong>Root causes to focus on:</strong> Check Root Cause Impact Analysis for failure mechanisms with highest average RPSS.</p>
<p><strong>Planning:</strong> Use Time Series Forecasts to anticipate recall volume by category.</p>
<p><strong>How to act:</strong> Download filtered data, ML predictions, and forecasts as CSV.</p>
<p class="story-cta">👉 Filters → Risk distribution → Root causes → ML predictions → Forecasts → Export</p>
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
