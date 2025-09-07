import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

# Paths
results_dir = Path("results")
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# Load selected features + labels
X = np.load(results_dir / "X_selected.npy")
y = np.load(results_dir / "y.npy")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest with tuned params
model = RandomForestClassifier(
    n_estimators=50, max_depth=None, min_samples_split=5, random_state=42
)
model.fit(X_train, y_train)

# Save model
model_path = models_dir / "heart_disease_model.pkl"
joblib.dump(model, model_path)

print(f"Model trained and saved to {model_path}")
