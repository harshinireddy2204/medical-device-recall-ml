# FDA Medical Device Recall Dashboard — Deployment Guide

> **Quick steps:** See [DEPLOY_STEPS.md](DEPLOY_STEPS.md) for exact commands and Git push instructions.

## Overview

You can deploy the dashboard in two ways:

| Option | Data Source | Best For |
|--------|-------------|----------|
| **Streamlit Cloud** | CSV snapshot | Public demo, no database required |
| **Self-hosted** | SQL Server | Full data, production use |

---

## Option 1: Streamlit Community Cloud (Recommended for Demo)

Deploy a public demo using CSV data—no SQL Server needed.

### Prerequisites

1. **GitHub account** and your repo pushed: [github.com/harshinireddy2204/medical-device-recall-ml](https://github.com/harshinireddy2204/medical-device-recall-ml)
2. **CSV snapshot** at `visualization/device_rpss_sample.csv` (see [Export CSV](#export-csv-from-sql-server) if you don't have it)

### Steps

1. **Create the CSV snapshot** (if needed)
   ```bash
   # From project root, with SQL Server running:
   python Scripts/export_device_rpss_to_csv.py
   ```
   This writes `visualization/device_rpss_sample.csv`.

2. **Commit and push** the CSV and code:
   ```bash
   git add visualization/device_rpss_sample.csv visualization/app_.py
   git commit -m "Add CSV snapshot for Streamlit Cloud deployment"
   git push origin main
   ```

3. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **"New app"**

4. **Configure the app**
   - **Repository:** `harshinireddy2204/medical-device-recall-ml` (or your fork)
   - **Branch:** `main`
   - **Main file path:** `visualization/app_.py`
   - **App URL:** e.g. `fda-recall-ml` → `https://fda-recall-ml.streamlit.app`

5. **Advanced settings** (optional)
   - Python version: 3.11
   - `requirements.txt` is auto-detected

6. **Deploy**  
   Click **Deploy**. The app will build and run. First load may take 1–2 minutes.

### Notes for Streamlit Cloud

- Uses **`app_.py`** with `device_rpss_sample.csv` (no database)
- ML predictions and time series forecasting work (scikit-learn in requirements)
- No `vw_FDA_Device_Integrated` or device names from DB (app_.py uses CSV only)

---

## Option 2: Self-Hosted (SQL Server)

For production with live data and full features (device names, integrated view).

### Prerequisites

- Python 3.11+
- SQL Server (local or remote) with FDADatabase
- ODBC Driver 17 for SQL Server

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt
pip install sqlalchemy pyodbc  # For database

# Launch full dashboard
streamlit run visualization/app.py
```

### Run on a Server (e.g. Azure VM, AWS EC2)

1. Install Python, SQL Server (or connect to remote), and ODBC driver.
2. Update connection string in `app.py` if SQL Server is remote:
   ```python
   # In app.py, modify get_engine():
   "SERVER=your-sql-server-host;"  # Instead of localhost
   ```
3. Run behind a reverse proxy (nginx) and use HTTPS.
4. Use `streamlit run visualization/app.py --server.port 8501 --server.address 0.0.0.0`

### Docker (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install sqlalchemy pyodbc

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "visualization/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t fda-recall-dashboard .
docker run -p 8501:8501 fda-recall-dashboard
```

---

## Export CSV from SQL Server

If `device_rpss_sample.csv` doesn’t exist, run:

```bash
python Scripts/export_device_rpss_to_csv.py
```

This script:

- Connects to `FDADatabase` on localhost
- Reads from `model.device_rpss`
- Writes `visualization/device_rpss_sample.csv` (up to 50,000 rows)

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| **"Could not read CSV file"** | Run `export_device_rpss_to_csv.py` and commit the CSV |
| **"ML modules not available"** | Ensure `scikit-learn` and `joblib` are in requirements.txt |
| **Streamlit Cloud build fails** | Check Python version (3.11 recommended) and requirements.txt |
| **Database connection failed** | Use `app_.py` for cloud; use `app.py` only with SQL Server |

---

## Summary

- **Quick public demo:** Push to GitHub → Streamlit Cloud → use `app_.py`
- **Production with full data:** Host SQL Server → use `app.py` locally or on a server
