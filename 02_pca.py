import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

# Paths
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Load preprocessed data
X_processed = np.load(results_dir / "X_processed.npy")
y = np.load(results_dir / "y.npy")

print("Data loaded for PCA")
print(f"X_processed shape: {X_processed.shape}, y shape: {y.shape}")

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_processed)

# Save transformed data
np.save(results_dir / "X_pca.npy", X_pca)
print(f"PCA complete. Transformed shape: {X_pca.shape}")

# Scree plot (explained variance ratio)
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Scree Plot")
plt.grid(True)
plt.savefig(results_dir / "pca_scree_plot.png")
plt.close()

print("PCA scree plot saved.")