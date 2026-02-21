# FDA Medical Device Recall Risk Intelligence System

## 🎯 Project Overview

An end-to-end data engineering and analytics pipeline that transforms fragmented FDA post-market surveillance data into actionable risk intelligence. The system processes over **2.4 billion adverse events** across **9,100+ medical devices** to predict recall severity patterns and enable risk-based regulatory prioritization.

**Live Dashboard:** [View Demo](#) *(Add your deployed URL here)*  

📘 **Deploy:** See [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) for Streamlit Cloud or self-hosted instructions.

![Dashboard Preview]
<img width="1913" height="975" alt="image" src="https://github.com/user-attachments/assets/f34a4e17-8012-4500-8905-1b9ea1b3aeb4" />


---

## 💡 Business Problem

**Challenge:** The FDA receives thousands of medical device recalls annually, but not all recalls represent equal risk. A device with one labeling recall from 5 years ago shouldn't receive the same regulatory attention as a device with multiple software failures causing patient harm.

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
│  Tables:                                    │
│  • dbo.MAUDE (adverse events)               │
│  • dbo.Premarket510k (clearances)           │
│  • dbo.recall (recall data)                 │
│  • dbo.Productcode (device classification)  │
│                                             │
│  Integrated View:                           │
│  • vw_FDA_Device_Integrated (29 columns)    │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│    RPSS Scoring Engine (rpss_pipeline.py)   │
│  • Aggregates data by device (PMA_PMN_NUM)  │
│  • Calculates 5 risk components             │
│  • Normalizes scores (0-1 scale)            │
│  • Categorizes: Low/Medium/High/Critical    │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│       Analytics Tables (model schema)       │
│  • model.device_rpss (scored devices)       │
│  • model.device_risk_scores (time-series)   │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│   Streamlit Dashboard (app_ultra_optimized) │
│  • Interactive filtering & visualization    │
│  • Real-time KPI metrics                    │
│  • Root cause impact analysis               │
│  • Export capabilities                      │
└─────────────────────────────────────────────┘
```

---

## 📊 Key Features

### Data Pipeline
- ✅ **Automated ETL**: Ingests 5 FDA data sources with incremental updates
- ✅ **Data Quality**: Handles duplicates, NULL values, and type mismatches
- ✅ **Scalability**: Processes 2.4B+ records using chunking and BIGINT optimization
- ✅ **Reproducibility**: Fully scripted pipeline from raw data to analytics

### RPSS Scoring Algorithm
- ✅ **Multi-dimensional Risk**: Combines frequency, severity, exposure, recency, and device class
- ✅ **Normalization**: Min-max scaling for fair comparison across metrics
- ✅ **Root Cause Mapping**: Assigns severity weights to 13+ failure mechanisms
- ✅ **Staging & MERGE**: Ensures idempotent, production-grade updates

### Interactive Dashboard
- ✅ **Performance Optimized**: SQLAlchemy with server-side aggregation for sub-2s queries
- ✅ **Advanced Filtering**: Multi-select filters with real-time KPI updates
- ✅ **Professional Visualizations**: Plotly charts with drill-down capabilities
- ✅ **Export Functionality**: CSV download of filtered datasets

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Sources** | FDA MAUDE, 510(k), PMA, Recall APIs | Medical device post-market data |
| **ETL Pipeline** | Python 3.13, PyODBC, Pandas | Data extraction and transformation |
| **Database** | SQL Server 2022 (LocalHost) | Structured data storage |
| **Scoring Engine** | Python, NumPy, SQLAlchemy | RPSS calculation and normalization |
| **Dashboard** | Streamlit, Plotly, Pandas | Interactive analytics interface |
| **Version Control** | Git, GitHub | Code management |

---

## 📁 Project Structure

```
FDA_pipeline/
├── data/
│   └── raw/                          # Raw FDA data files
│       ├── 510k/                     # Premarket clearances
│       ├── MAUDE/                    # Adverse events
│       ├── PMA/                      # Premarket approvals
│       ├── prodclass/                # Product classifications
│       └── recall/                   # Recall data
│
├── Scripts/                          # Python ETL scripts
│   ├── load_maude_pyodbc_pipeline.py # MAUDE data loader
│   ├── load_pma510kprocode.py       # PMA/510k/Product code loader
│   ├── load_recall.py               # Recall data loader
│   └── rpss_pipeline.py             # RPSS scoring engine
│
├── sql/
│   ├── create_tables.sql            # Database schema
│   ├── create_integrated_view.sql   # Data integration view
│   └── create_enhanced_view.sql     # Dashboard-ready view
│
├── visualization/
│   ├── app_ultra_optimized.py       # Production dashboard
│   └── assets/                      # Dashboard images/logos
│
├── docs/
│   ├── DATA_DICTIONARY.md           # Column descriptions
│   ├── RPSS_METHODOLOGY.md          # Scoring algorithm details
│   └── DEPLOYMENT_GUIDE.md          # Setup instructions
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- SQL Server 2019+ (or SQL Server Express)
- ODBC Driver 17 for SQL Server
- Git

### Installation

1. **Clone the repository**
   
   git clone https://github.com/harshinireddy2204/medical-device-recall-ml.git
   cd medical-device-recall-ml
  

2. **Install Python dependencies**
   
   pip install -r requirements.txt
   

3. **Set up SQL Server database**
   
   # Create database
   sqlcmd -S localhost -Q "CREATE DATABASE FDADatabase"
   
   # Run schema scripts
   sqlcmd -S localhost -d FDADatabase -i sql/create_tables.sql
   

4. **Download FDA data** *(See [Data Sources](#data-sources))*
   - Place files in `data/raw/` folders

5. **Run ETL pipeline**
   python Scripts/load_maude_pyodbc_pipeline.py
   python Scripts/load_pma510kprocode.py
   python Scripts/load_recall.py
   

6. **Generate RPSS scores**
   python Scripts/rpss_pipeline.py
   

7. **Launch dashboard**
   streamlit run visualization/app_ultra_optimized.py
   

## 📊 Sample Results

### Risk Distribution
- **Low Risk:** 7,200 devices (79%)
- **Medium Risk:** 1,400 devices (15%)
- **High Risk:** 350 devices (4%)
- **Critical Risk:** 163 devices (2%)

**Key Finding:** Top 2% of devices (Critical) account for 60%+ of high-severity recalls and adverse events.

### Root Cause Analysis
| Root Cause      | Avg RPSS  | Device Count   | Total Recalls |
|-----------------|---------- |--------------  |---------------|
| Software Design | 0.87      | 245            | 1,230         |
| Device Design   | 0.81      | 412            | 1,890         |
| Process Control | 0.74      | 328            | 1,450         |
| Manufacturing   | 0.68      | 520            | 2,100         |

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

All data sourced from publicly available FDA databases:

- **MAUDE:** [FDA Medical Device Adverse Events](https://www.fda.gov/medical-devices/mandatory-reporting-requirements-manufacturers-importers-and-device-user-facilities/medical-device-reporting-mdr-how-report-medical-device-problems)
- **510(k):** [Premarket Notification Database](https://www.fda.gov/medical-devices/device-approvals-denials-and-clearances/510k-clearances)
- **PMA:** [Premarket Approval Database](https://www.fda.gov/medical-devices/device-approvals-denials-and-clearances/pma-approvals)
- **Recalls:** [Medical Device Recalls Database](https://www.fda.gov/medical-devices/medical-device-recalls)

---

## 👤 Author

**Harshini Reddy**  
Business and Data Analyst
📧 Email: harshini.dommata@gmail.com  
💼 LinkedIn: https://www.linkedin.com/in/harshini-reddy22/

## 🙏 Acknowledgments

- FDA for providing open-access medical device data
- Streamlit community for visualization framework
- SQLAlchemy team for database abstraction layer

## 📞 Contact

For questions, collaboration opportunities, or feedback:
- Open an issue in this repository
- Email: harshini.dommata@gmail.com
- LinkedIn: https://www.linkedin.com/in/harshini-reddy22/

**⭐ If you found this project useful, please consider giving it a star!**
