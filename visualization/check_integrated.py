import pyodbc
import pandas as pd

# Connect to SQL Server
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=FDADatabase;"
    "Trusted_Connection=yes;"
)

print("=" * 80)
print("CHECKING SOURCE VIEW: dbo.vw_FDA_Device_Integrated")
print("=" * 80)

cursor = conn.cursor()

# Check if the integrated view exists
cursor.execute("""
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.VIEWS 
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'vw_FDA_Device_Integrated'
""")

view_exists = cursor.fetchone()

if view_exists:
    print("✅ View exists: dbo.vw_FDA_Device_Integrated\n")
    
    # Get all columns
    print("COLUMNS IN VIEW:")
    print("-" * 80)
    cursor.execute("""
        SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'vw_FDA_Device_Integrated'
        ORDER BY ORDINAL_POSITION
    """)
    
    columns = cursor.fetchall()
    for i, col in enumerate(columns, 1):
        col_name, data_type, max_length = col
        length_info = f"({max_length})" if max_length else ""
        print(f"{i:2d}. {col_name:40s} {data_type}{length_info}")
    
    print("\n" + "=" * 80)
    print(f"TOTAL COLUMNS: {len(columns)}")
    print("=" * 80)
    
    # Get sample data
    print("\nSAMPLE DATA (first 3 rows):")
    print("-" * 80)
    try:
        df = pd.read_sql("SELECT TOP 3 * FROM dbo.vw_FDA_Device_Integrated", conn)
        print(f"Successfully loaded {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        print("\nPreview:")
        print(df.head(3).to_string())
    except Exception as e:
        print(f"❌ Error: {str(e)}")
else:
    print("❌ View does NOT exist!")

print("\n" + "=" * 80)
print("COMPARISON: What columns are used by RPSS pipeline?")
print("=" * 80)

print("""
From rpss_pipeline.py, the pipeline uses these columns:
1. PMA_PMN_NUM (device identifier)
2. cfres_id (for recall count)
3. adverse_event_count (for total adverse events)
4. unique_manufacturers
5. unique_distributors
6. DEVICECLASS (device class)
7. first_event_date
8. last_event_date
9. root_cause_description

The pipeline aggregates these to create the RPSS scores.
""")

print("\n" + "=" * 80)
print("RECOMMENDATION FOR DASHBOARD VIEW")
print("=" * 80)

print("""
The dashboard view (vw_device_rpss_categorized) should include:

FROM model.device_rpss table:
- PMA_PMN_NUM
- rpss
- rpss_category
- recall_count
- total_adverse_events  ⚠️ MISSING in current view
- unique_manufacturers  ⚠️ MISSING in current view
- device_class
- root_cause_description
- last_scored           ⚠️ MISSING in current view

PLUS additional enrichment from vw_FDA_Device_Integrated (OPTIONAL):
- Device names/descriptions
- Manufacturer names
- Product codes
- FDA decision dates
- etc.

This would give the dashboard more context for filtering and display.
""")

conn.close()