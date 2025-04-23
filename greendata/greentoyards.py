import pandas as pd

# Load filtered feet data
df = pd.read_csv("data/green_under90_feet.csv")

# Convert holedis (feet → yards)
df["holedis"] = df["holedis"] / 3

# Save as yard-based data
df.to_csv("data/green_under30_yards.csv", index=False)
print("Saved green data converted to yards (max 30 yds)")
