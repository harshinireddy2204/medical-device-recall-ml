# FDA Medical Device Recall Dashboard — Step-by-Step Deployment

## Overview

- **app.py** — Uses local SQL Server. Works on localhost only. Cannot be deployed to Streamlit Cloud (no access to your local DB).
- **app_.py** — Uses CSV snapshot. Works on Streamlit Cloud. Use this for public deployment.

---

## Step 1: Export CSV (with device names)

On your machine, with SQL Server running and `model.device_rpss` populated:

```bash
cd C:\Users\harsh\FDA_pipeline\medical-device-recall-ml

# Install dependencies if needed
pip install sqlalchemy pyodbc pandas

# Run export script
python Scripts/export_device_rpss_to_csv.py
```

**Output:** `visualization/device_rpss_sample.csv` (up to 50,000 rows, includes `device_name` from `vw_FDA_Device_Integrated` if the view exists).

**Fallback:** If `vw_FDA_Device_Integrated` does not exist, the script exports without `device_name` (uses "—" placeholder).

---

## Step 2: Verify CSV and local app

Check the CSV:

```bash
# Quick check (first few lines)
head -5 visualization/device_rpss_sample.csv
```

Test app_.py locally:

```bash
streamlit run visualization/app_.py
```

Confirm:
- KPIs load
- Risk distribution chart shows
- Root cause analysis shows
- ML predictions run (click Generate ML Predictions)
- Time series forecasts run (click Generate Forecasts)
- Device names appear in ML Top 20 (if CSV has `device_name`)

---

## Step 3: Commit and push to GitHub

Only push what’s needed for Streamlit Cloud:

```bash
cd C:\Users\harsh\FDA_pipeline\medical-device-recall-ml

# Stage required files
git add visualization/app_.py
git add visualization/device_rpss_sample.csv
git add Scripts/export_device_rpss_to_csv.py
git add Scripts/ml_recall_prediction.py
git add Scripts/time_series_forecast.py
git add requirements.txt
git add docs/DEPLOY_STEPS.md
git add docs/DEPLOYMENT_GUIDE.md
git add README.md

# Optional (not required for Streamlit Cloud)
git add visualization/app.py
git add Scripts/rpss_pipeline.py
git add sql/

# Commit
git commit -m "Add Streamlit Cloud deployment: app_.py with CSV, device names, story arc"

# Push
git push origin main
```

**Important:** `device_rpss_sample.csv` is explicitly allowed in `.gitignore` via `!visualization/device_rpss_sample.csv`, so it will be tracked.

---

## Step 4: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Set:
   - **Repository:** `harshinireddy2204/medical-device-recall-ml` (or your repo)
   - **Branch:** `main`
   - **Main file path:** `visualization/app_.py`
5. Click **Deploy**

Initial build takes ~1–2 minutes.

---

## Step 5: After deployment

- App URL: `https://<app-name>.streamlit.app`
- Update `README.md` with your live URL
- To refresh data, run `python Scripts/export_device_rpss_to_csv.py`, commit and push the new CSV, then trigger a redeploy in Streamlit Cloud (or push a small change)

---

## Files to push (checklist)

| File | Purpose |
|------|---------|
| `visualization/app_.py` | Cloud dashboard |
| `visualization/device_rpss_sample.csv` | Data snapshot |
| `Scripts/export_device_rpss_to_csv.py` | CSV export script |
| `Scripts/ml_recall_prediction.py` | ML module |
| `Scripts/time_series_forecast.py` | Time series module |
| `requirements.txt` | Dependencies |
| `docs/DEPLOY_STEPS.md` | Deployment steps |
| `docs/DEPLOYMENT_GUIDE.md` | Deployment guide |
| `README.md` | Project overview |
| `.gitignore` | Ignore rules (keep `!visualization/device_rpss_sample.csv`) |

---

## CSV data coverage

| Column | Source | Notes |
|--------|--------|-------|
| PMA_PMN_NUM | model.device_rpss | Device ID |
| rpss | model.device_rpss | Risk score |
| rpss_category | model.device_rpss | Low/Medium/High/Critical |
| recall_count | model.device_rpss | Number of recalls |
| total_adverse_events | model.device_rpss | Adverse event count |
| unique-manufacturers | model.device_rpss | Manufacturer count |
| device_class | model.device_rpss | FDA class 1/2/3 |
| root_cause_description | model.device_rpss | Root cause |
| last_scored | model.device_rpss | Score date |
| device_name | vw_FDA_Device_Integrated | Device name (k_devicename, pc_devicename, GENERICNAME, TRADENAME) |

---

## Deploying app.py instead

To deploy the SQL-backed app (app.py), you would need:

1. A hosted SQL Server (Azure SQL, AWS RDS, etc.)
2. Migration of `FDADatabase` and views
3. Connection string in app.py (via Streamlit secrets)
4. Credentials and network access from Streamlit Cloud

That is more involved. For a public demo, the CSV-based app_.py is recommended.
