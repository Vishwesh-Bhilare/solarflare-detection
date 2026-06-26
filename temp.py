import pandas as pd
df = pd.read_parquet("data/processed/smoothed_data.parquet")
print(df.shape)
print(df.head())
