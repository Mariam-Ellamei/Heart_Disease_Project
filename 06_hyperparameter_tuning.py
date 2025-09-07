import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# Paths
results_dir = Path("results")

# Load selected features + labels
X = np.load(results_dir / "X_selected.npy")
y = np.load(results_dir / "y.npy")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Starting Random Forest hyperparameter tuning: ")

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10]
}

# GridSearchCV
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train, y_train)

print("Best parameters found:")
print(grid.best_params_)
print(f"Best CV accuracy: {grid.best_score_:.4f}")

# Evaluate on test set
test_acc = grid.score(X_test, y_test)
print(f"Test set accuracy with best model: {test_acc:.4f}")
