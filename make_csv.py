import pandas as pd
from pathlib import Path

raw_path = Path("data/processed.cleveland.data")
out_path = Path("data/heart_disease.csv")

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

# Load the raw file
df = pd.read_csv(raw_path, header=None, names=columns)

# Replace "?" with NaN and drop missing values
df = df.replace("?", pd.NA).dropna()

# Convert all columns to numeric
df = df.apply(pd.to_numeric)

# Convert target 'num' into binary: 0 = no disease, 1 = disease
df["target"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num"])

# Save as CSV
df.to_csv(out_path, index=False)

print(f"Saved cleaned dataset to {out_path}, shape = {df.shape}")
print(df.head())

