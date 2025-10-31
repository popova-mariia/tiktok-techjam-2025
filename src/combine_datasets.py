import pandas as pd

# Load both datasets
df1 = pd.read_csv("../data/reviews_original.csv")
df2 = pd.read_csv("../data/reviews_from_russia.csv")

# Combine them
combined = pd.concat([df1, df2], ignore_index=True)

# Save
combined.to_csv("../data/reviews_combined.csv", index=False)
print(f"Saved combined dataset: {len(combined)} rows")
