import pandas as pd

# Load full green dataset
df = pd.read_csv("data/shots_under60_green.csv")  # or your original green file

# Filter to distances ≤ 90 feet
df_filtered = df[df["holedis"] <= 90].copy()

# Save filtered version
df_filtered.to_csv("data/green_under90_feet.csv", index=False)
print("Saved filtered green data (≤ 90 feet)")
