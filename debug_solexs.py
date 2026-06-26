import pandas as pd

# Read the parquet file
df = pd.read_parquet("data/processed/solexs.parquet")

# Check timestamp range
print("Timestamp range:")
print(f"  Min: {df['timestamp'].min()}")
print(f"  Max: {df['timestamp'].max()}")
print()

# Check if June 11 exists
print("Checking for June 11, 2024 data:")
june_11_data = df[
    (df["timestamp"] >= "2024-06-11") &
    (df["timestamp"] < "2024-06-12")
]

print(f"  Number of rows on June 11: {len(june_11_data)}")

if len(june_11_data) > 0:
    print("\n  First 5 rows from June 11:")
    print(june_11_data.head())
    print("\n  Last 5 rows from June 11:")
    print(june_11_data.tail())
else:
    print("  No data found for June 11, 2024")
    print("\n  Checking dates around that period:")
    
    # Check what dates are available
    df['date_only'] = df['timestamp'].dt.date
    
    # Show unique dates
    print("\n  Unique dates in the dataset (first 20):")
    unique_dates = df['date_only'].unique()
    print(sorted(unique_dates)[:20])
    
    # Check for any data in June 2024
    june_2024_data = df[
        (df["timestamp"] >= "2024-06-01") &
        (df["timestamp"] < "2024-07-01")
    ]
    print(f"\n  Total rows in June 2024: {len(june_2024_data)}")
    if len(june_2024_data) > 0:
        print(f"  Date range in June: {june_2024_data['timestamp'].min()} to {june_2024_data['timestamp'].max()}")
        print("\n  Unique dates in June 2024:")
        print(sorted(june_2024_data['date_only'].unique())[:20])
