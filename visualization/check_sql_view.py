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
print("CHECKING SQL VIEW: dbo.vw_device_rpss_categorized")
print("=" * 80)

# Check if view exists
cursor = conn.cursor()
cursor.execute("""
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.VIEWS 
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'vw_device_rpss_categorized'
""")

view_exists = cursor.fetchone()

if view_exists:
    print("✅ View exists: dbo.vw_device_rpss_categorized\n")
    
    # Get column information
    print("COLUMNS IN VIEW:")
    print("-" * 80)
    cursor.execute("""
        SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'vw_device_rpss_categorized'
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
    
    # Try to fetch sample data
    print("\nFETCHING SAMPLE DATA (first 5 rows)...")
    print("-" * 80)
    try:
        df = pd.read_sql("SELECT TOP 5 * FROM dbo.vw_device_rpss_categorized", conn)
        print(f"\nSuccessfully loaded {len(df)} sample rows")
        print("\nColumn names from pandas:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        print("\nSample data preview:")
        print(df.head())
        
    except Exception as e:
        print(f"❌ Error fetching data: {str(e)}")
    
else:
    print("❌ View does NOT exist: dbo.vw_device_rpss_categorized")
    print("\nLooking for similar views...")
    cursor.execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME 
        FROM INFORMATION_SCHEMA.VIEWS 
        WHERE TABLE_NAME LIKE '%rpss%' OR TABLE_NAME LIKE '%device%'
        ORDER BY TABLE_SCHEMA, TABLE_NAME
    """)
    
    similar_views = cursor.fetchall()
    if similar_views:
        print("\nFound these views:")
        for schema, name in similar_views:
            print(f"  - {schema}.{name}")
    else:
        print("No similar views found.")

# Check model.device_rpss table
print("\n" + "=" * 80)
print("CHECKING TABLE: model.device_rpss")
print("=" * 80)

cursor.execute("""
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_SCHEMA = 'model' AND TABLE_NAME = 'device_rpss'
""")

table_exists = cursor.fetchone()

if table_exists:
    print("✅ Table exists: model.device_rpss\n")
    
    print("COLUMNS IN TABLE:")
    print("-" * 80)
    cursor.execute("""
        SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'model' AND TABLE_NAME = 'device_rpss'
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
    
    # Sample data
    try:
        df_model = pd.read_sql("SELECT TOP 5 * FROM model.device_rpss", conn)
        print(f"\nSuccessfully loaded {len(df_model)} sample rows from model.device_rpss")
        print("\nSample data:")
        print(df_model.head())
    except Exception as e:
        print(f"❌ Error fetching data: {str(e)}")
else:
    print("❌ Table does NOT exist: model.device_rpss")

conn.close()

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)