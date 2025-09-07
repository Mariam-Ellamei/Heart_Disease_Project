import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# Paths
data_path = Path("data/heart_disease.csv")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Load dataset
print("🔹 Loading dataset...")
df = pd.read_csv(data_path)
print(f"Shape: {df.shape}")
print(df.head())

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# Preprocessing: scale numeric features (all are numeric here)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features)
    ],
    remainder="passthrough"
)

# Build pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# Fit and transform
X_processed = pipeline.fit_transform(X)

# Save processed data and preprocessor
np.save(results_dir / "X_processed.npy", X_processed)
np.save(results_dir / "y.npy", y)
joblib.dump(pipeline, results_dir / "preprocessor.pkl")

print("Preprocessing complete. Saved files in 'results/' folder.")

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(results_dir / "correlation_heatmap.png")
plt.close()
print("Correlation heatmap saved.")
