# FDA Medical Device Recall Risk Intelligence System

## 🎯 Project Overview

An end-to-end data engineering and analytics pipeline that transforms fragmented FDA post-market surveillance data into actionable risk intelligence. The system processes over **2.4 billion adverse events** across **9,100+ medical devices** to predict recall severity patterns and enable risk-based regulatory prioritization.

**Live Dashboard:** [View Demo](#) *(Replace with your deployed Streamlit Cloud URL after deployment)*

---

## 💡 Business Problem

**Challenge:** The FDA receives thousands of medical device recalls annually, but not all recalls represent equal risk. A device with one labeling recall from five years ago shouldn't receive the same regulatory attention as a device with multiple software failures causing patient harm.

**Solution:** The **Recall Pattern Severity Score (RPSS)** consolidates five risk dimensions into a single predictive metric:
- Recall frequency (30%)
- Root cause severity (30%)
- Adverse event exposure (20%)
- Temporal recency (10%)
- Device classification (10%)

**Impact:** Enables regulators and manufacturers to:
- Identify the top 2% of critical devices driving 60%+ of high-severity recalls
- Prioritize quality system audits based on failure mechanism patterns
- Allocate post-market surveillance resources where risk is highest

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  FDA Data APIs  │
│  - MAUDE        │
│  - 510(k)       │
│  - PMA          │
│  - Recalls      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│       Python ETL Pipeline (PyODBC)          │
│  • Data extraction from FDA sources         │
│  • Data cleaning & deduplication            │
│  • Type validation & normalization          │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│    SQL Server Database (LocalHost)          │
│  Tables: MAUDE, Premarket510k, recall,      │
│  Productcode; View: vw_FDA_Device_Integrated│
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│    RPSS Scoring Engine (rpss_pipeline.py)   │
│  • Aggregates by device (PMA_PMN_NUM)       │
│  • Calculates 5 risk components             │
│  • Categorizes: Low/Medium/High/Critical    │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│       Streamlit Dashboards                  │
│  • app.py   — Local (SQL Server)            │
│  • app_.py  — Cloud (CSV snapshot)          │
└─────────────────────────────────────────────┘
```

---

## 📊 Key Features

### Dashboard
- **Local (app.py):** Full SQL Server data, device names from integrated view, ML predictions, time series forecasting
- **Cloud (app_.py):** CSV-based demo for Streamlit Cloud—no database required
- **Story-driven UX:** 4-step journey from risk overview → root causes → ML predictions → forecasts
- **ML & Forecasting:** Recall likelihood prediction (Random Forest), time series forecasting by category
- **Export:** CSV download for filtered data, predictions, and forecasts

### Data Pipeline
- Automated ETL from 5 FDA data sources
- RPSS scoring with 13+ root cause mappings
- BIGINT-safe aggregation for large datasets

---

## 📁 Project Structure

```
medical-device-recall-ml/
├── Scripts/
│   ├── export_device_rpss_to_csv.py   # Export for Streamlit Cloud
│   ├── load_maude_pyodbc_pipeline.py
│   ├── load_pma510kprocode.py
│   ├── load_recall.py
│   ├── rpss_pipeline.py
│   ├── ml_recall_prediction.py
│   └── time_series_forecast.py
├── sql/
│   ├── create_tables.sql
│   ├── create_integrated_view.sql
│   └── create_enhanced_view.sql
├── visualization/
│   ├── app.py          # Local dashboard (SQL Server)
│   ├── app_.py         # Cloud dashboard (CSV)
│   └── device_rpss_sample.csv
├── docs/
│   ├── DEPLOY_STEPS.md
│   └── DEPLOYMENT_GUIDE.md
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- SQL Server 2019+ (or Express) — for local dashboard
- ODBC Driver 17 for SQL Server
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/medical-device-recall-ml.git
cd medical-device-recall-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For the local dashboard (app.py), also install:

```bash
pip install sqlalchemy pyodbc
```

### 3. Run locally

**Option A — Local dashboard (SQL Server required)**

```bash
# Ensure FDADatabase and model.device_rpss exist
streamlit run visualization/app.py
```

**Option B — Cloud-style dashboard (CSV only)**

```bash
# Uses visualization/device_rpss_sample.csv
streamlit run visualization/app_.py
```

---

## 📤 Deployment

### Streamlit Cloud (public demo)

Use `app_.py` with the CSV snapshot. No SQL Server needed.

1. **Export CSV** (if you have SQL Server and want fresh data):

   ```bash
   pip install sqlalchemy pyodbc pandas
   python Scripts/export_device_rpss_to_csv.py
   ```

2. **Push to GitHub** and deploy:

   See [docs/DEPLOY_STEPS.md](docs/DEPLOY_STEPS.md) for step-by-step instructions.

3. **Deploy on Streamlit Cloud**

   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - New app → Repository: `YOUR_USERNAME/medical-device-recall-ml`, Main file: `visualization/app_.py`
   - Deploy

**Full deployment guide:** [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)

---

## 📊 Sample Results

### Risk Distribution
- **Low Risk:** ~79% of devices
- **Medium Risk:** ~15%
- **High Risk:** ~4%
- **Critical Risk:** ~2%

**Finding:** Top 2% (Critical) account for 60%+ of high-severity recalls and adverse events.

### Root Cause Analysis (example)
| Root Cause      | Avg RPSS | Devices |
|-----------------|----------|---------|
| Software Design | 0.87     | 245     |
| Device Design   | 0.81     | 412     |
| Process Control | 0.74     | 328     |

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

**Data Engineering:**
- ETL pipeline design and implementation
- Database schema design and optimization
- Handling large-scale datasets (2B+ records)
- Data quality assurance and validation

**Analytics & ML:**
- Multi-dimensional scoring algorithm development
- Statistical normalization techniques
- Root cause pattern analysis
- Predictive risk modeling

**Software Engineering:**
- Production-grade Python development
- SQL query optimization
- Version control with Git
- Code documentation and testing

**Data Visualization:**
- Interactive dashboard development
- Stakeholder-focused storytelling
- Performance optimization for large datasets

---

## 📈 Future Enhancements

- [ ] **NLP Analysis:** Extract insights from recall reason descriptions
- [ ] **API Development:** REST API for external system integration
- [ ] **Cloud Deployment:** Deploy dashboard on AWS/Azure for public access
- [ ] **Automated Alerts:** Email notifications for new critical-risk devices

---

## 📝 Data Sources

- [MAUDE](https://www.fda.gov/medical-devices/mandatory-reporting-requirements-manufacturers-importers-and-device-user-facilities/medical-device-reporting-mdr-how-report-medical-device-problems)
- [510(k)](https://www.fda.gov/medical-devices/device-approvals-denials-and-clearances/510k-clearances)
- [PMA](https://www.fda.gov/medical-devices/device-approvals-denials-and-clearances/pma-approvals)
- [Recalls](https://www.fda.gov/medical-devices/medical-device-recalls)

---

## 🙏 Acknowledgments

- FDA for open-access medical device data
- Streamlit and SQLAlchemy communities

---

**⭐ If you found this project useful, please consider giving it a star!**
