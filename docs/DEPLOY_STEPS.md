# FDA Medical Device Recall Dashboard — Step-by-Step Deployment

This guide walks you through deploying the dashboard to Streamlit Cloud so others can use it without a local database.

---

## Overview

| App    | Data Source  | Use Case                         |
|--------|--------------|----------------------------------|
| `app.py`  | SQL Server   | Local use with full data         |
| `app_.py` | CSV snapshot | Streamlit Cloud (public demo)    |

For Streamlit Cloud, use **`app_.py`** because it loads from a CSV file—no database connection needed.

---

## Step 1: Export the CSV (if you have SQL Server)

If you have SQL Server with `model.device_rpss` populated and want fresh data:

1. Open a terminal in the project root folder.
2. Install dependencies (if needed):

   ```bash
   pip install sqlalchemy pyodbc pandas
   ```

3. Run the export script:

   ```bash
   python Scripts/export_device_rpss_to_csv.py
   ```

**Output:** `visualization/device_rpss_sample.csv` (up to 50,000 rows, includes `device_name` if `vw_FDA_Device_Integrated` exists).

If the view does not exist, the script still runs and uses "—" for device names.

**Note:** If you already have a valid `device_rpss_sample.csv` in the repo, you can skip this step.

---

## Step 2: Test the app locally

Run the cloud-style app to confirm it works:

```bash
streamlit run visualization/app_.py
```

Check that:

- KPIs load
- Risk distribution chart displays
- Root cause analysis shows
- ML predictions run (click **Generate ML Predictions**)
- Time series forecasts run (click **Generate Forecasts**)
- Device names appear in ML Top 20 (if the CSV includes `device_name`)

---

## Step 3: Push to GitHub

From the project root:

1. Stage the required files:

   ```bash
   git add visualization/app_.py
   git add visualization/device_rpss_sample.csv
   git add Scripts/export_device_rpss_to_csv.py
   git add Scripts/ml_recall_prediction.py
   git add Scripts/time_series_forecast.py
   git add requirements.txt
   git add docs/DEPLOY_STEPS.md
   git add docs/DEPLOYMENT_GUIDE.md
   git add README.md
   ```

2. Optionally include local/ETL files:

   ```bash
   git add visualization/app.py
   git add Scripts/rpss_pipeline.py
   git add sql/
   ```

3. Commit and push:

   ```bash
   git commit -m "Add Streamlit Cloud deployment"
   git push origin main
   ```

**Note:** `device_rpss_sample.csv` is explicitly tracked via `!visualization/device_rpss_sample.csv` in `.gitignore`.

---

## Step 4: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Sign in with GitHub.
3. Click **New app**.
4. Set:
   - **Repository:** `YOUR_USERNAME/medical-device-recall-ml`
   - **Branch:** `main`
   - **Main file path:** `visualization/app_.py`
5. Click **Deploy**.

The first build may take 1–2 minutes. When it finishes, the app URL will look like:

`https://YOUR-APP-NAME.streamlit.app`

---

## Step 5: Update your README

After deployment:

1. Copy the live URL from Streamlit Cloud.
2. Update `README.md` and replace the placeholder in **Live Dashboard** with that URL.
3. Commit and push:

   ```bash
   git add README.md
   git commit -m "Update live dashboard URL"
   git push origin main
   ```

---

## Files checklist

| File                              | Purpose                |
|-----------------------------------|------------------------|
| `visualization/app_.py`           | Cloud dashboard        |
| `visualization/device_rpss_sample.csv` | Data snapshot     |
| `Scripts/export_device_rpss_to_csv.py` | CSV export script |
| `Scripts/ml_recall_prediction.py` | ML module              |
| `Scripts/time_series_forecast.py` | Time series module     |
| `requirements.txt`                | Dependencies           |
| `docs/DEPLOY_STEPS.md`            | This guide             |
| `docs/DEPLOYMENT_GUIDE.md`        | Extended deployment    |
| `README.md`                       | Project overview       |

---

## Pushing documentation updates

If you have edited `README.md`, `DEPLOY_STEPS.md`, or `DEPLOYMENT_GUIDE.md`, push the changes with:

```bash
git add README.md docs/DEPLOY_STEPS.md docs/DEPLOYMENT_GUIDE.md
git commit -m "Update README and deployment docs"
git push origin main
```

Replace `main` with your default branch if different.

---

## Refreshing data

To refresh the dashboard data:

1. Run `python Scripts/export_device_rpss_to_csv.py`.
2. Commit and push the new CSV.
3. Streamlit Cloud will redeploy on the next push, or you can trigger a redeploy from the Streamlit Cloud dashboard.
