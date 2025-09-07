import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

# Paths
results_dir = Path("results")

# Load preprocessed data (scaled features)
X = np.load(results_dir / "X_processed.npy")

print("Running KMeans clustering...")

# Try 2–6 clusters
scores = {}
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    scores[k] = score
    print(f"k={k}, silhouette score={score:.4f}")

# Plot silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(list(scores.keys()), list(scores.values()), marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("KMeans Clustering")
plt.grid(True)
plt.savefig(results_dir / "kmeans_silhouette.png")
plt.close()

print("Silhouette score plot saved.")
