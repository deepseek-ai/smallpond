"""
This script demonstrates the basic usage of Smallpond for processing Parquet files.
===================================
It shows how to:
- Read data from Parquet files
- Repartition data based on a column
- Apply SQL transformations
- Write results back to Parquet format
"""

import smallpond

# Initialize session
sp = smallpond.init()

# Load data
df = sp.read_parquet("prices.parquet")
print(df.to_pandas())

# Process data
df = df.repartition(3, hash_by="ticker")
df = sp.partial_sql("SELECT ticker, min(price), max(price) FROM {0} GROUP BY ticker", df)

# Save results
df.write_parquet("output/")
# Show results
print(df.to_pandas())