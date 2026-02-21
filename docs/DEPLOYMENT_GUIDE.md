# FDA Medical Device Recall Dashboard — Deployment Guide

> **Quick steps:** See [DEPLOY_STEPS.md](DEPLOY_STEPS.md) for step-by-step instructions and Git commands.

---

## Overview

You can run the dashboard in two ways:

| Option             | Data Source | Best For                       |
|--------------------|-------------|--------------------------------|
| **Streamlit Cloud**| CSV snapshot| Public demo, no database       |
| **Self-hosted**    | SQL Server  | Full data, production use      |

---

## Option 1: Streamlit Community Cloud (public demo)

Deploy a public demo using CSV data—no SQL Server required.

### Prerequisites

- GitHub account
- A fork or clone of this repository on GitHub
- `visualization/device_rpss_sample.csv` in the repo (see [Export CSV](#export-csv-from-sql-server) if you need to create it)

### Steps

1. **Create the CSV snapshot** (if needed)

   From the project root, with SQL Server running:

   ```bash
   python Scripts/export_device_rpss_to_csv.py
   ```

   This writes `visualization/device_rpss_sample.csv`.

2. **Commit and push** the CSV and app:

   ```bash
   git add visualization/device_rpss_sample.csv visualization/app_.py
   git commit -m "Add CSV snapshot for Streamlit Cloud deployment"
   git push origin main
   ```

3. **Deploy on Streamlit Cloud**

   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **New app**
   - Set:
     - **Repository:** `YOUR_USERNAME/medical-device-recall-ml`
     - **Branch:** `main`
     - **Main file path:** `visualization/app_.py`
   - Click **Deploy**

### What works on Streamlit Cloud

- Uses `app_.py` with `device_rpss_sample.csv` (no database)
- ML predictions and time series forecasting
- Device names if the CSV includes `device_name` from the export script

---

## Option 2: Self-hosted (SQL Server)

For production use with live data and full features.

### Prerequisites

- Python 3.11+
- SQL Server with `FDADatabase` and `model.device_rpss`
- ODBC Driver 17 for SQL Server

### Run locally

```bash
pip install -r requirements.txt
pip install sqlalchemy pyodbc

streamlit run visualization/app.py
```

### Run on a server (e.g. Azure VM, AWS EC2)

1. Install Python, SQL Server (or a remote connection), and the ODBC driver.
2. If using a remote SQL Server, update the connection in `app.py`:

   ```python
   # In get_engine(), change:
   "SERVER=your-sql-server-host;"  # instead of localhost
   ```

3. Run Streamlit on the desired port:

   ```bash
   streamlit run visualization/app.py --server.port 8501 --server.address 0.0.0.0
   ```

4. Use a reverse proxy (e.g. nginx) and HTTPS in production.

### Docker (optional)

Example `Dockerfile`:

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

To create or refresh `device_rpss_sample.csv`:

```bash
python Scripts/export_device_rpss_to_csv.py
```

This script:

- Connects to `FDADatabase` on localhost
- Reads from `model.device_rpss`
- Joins with `vw_FDA_Device_Integrated` for device names (if available)
- Writes `visualization/device_rpss_sample.csv` (up to 50,000 rows)

---

## Troubleshooting

| Issue                             | Fix                                                              |
|----------------------------------|------------------------------------------------------------------|
| "Could not read CSV file"        | Run `export_device_rpss_to_csv.py` and commit the CSV            |
| "ML modules not available"       | Add `scikit-learn` and `joblib` to `requirements.txt`            |
| Streamlit Cloud build fails      | Use Python 3.11 and verify `requirements.txt`                    |
| Database connection failed       | Use `app_.py` for cloud; use `app.py` only with SQL Server       |

---

## Summary

- **Public demo:** Push to GitHub → Streamlit Cloud → `app_.py`
- **Full data:** Host SQL Server → run `app.py` locally or on a server
