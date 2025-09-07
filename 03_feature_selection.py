import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from pathlib import Path

# Paths
data_path = Path("data/heart_disease.csv")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(data_path)

X = df.drop(columns=["target"])
y = df["target"]

# Feature selection: top 8 features using ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=8)
X_new = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()

# Save results
np.save(results_dir / "X_selected.npy", X_new)
pd.Series(selected_features).to_csv(results_dir / "selected_features.csv", index=False)

print("Feature selection complete.")
print(f"Selected features: {selected_features}")
