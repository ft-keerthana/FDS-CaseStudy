"""
Online Retail Transaction Analysis Using Clustering Algorithms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    SpectralClustering
)
from sklearn.metrics import silhouette_score, davies_bouldin_score

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "Online retail_new (1).csv"

print("Libraries imported successfully!\n")

# 1. Load Dataset using Configurable Local Path
raw_df = pd.read_csv(DATA_PATH)

print("Dataset shape:", raw_df.shape)
print(raw_df.head().to_string())

# 2. Preprocess Transactions
transactions = []
for _, row in raw_df.iterrows():
    basket = [
        str(item).strip()
        for item in row.tolist()
        if pd.notna(item) and str(item).strip() != ""
    ]
    if basket:
        transactions.append(basket)

print("\nTotal transactions:", len(transactions))
print("Total product occurrences:", sum(len(b) for b in transactions))
print("Unique products:", len(set(item for b in transactions for item in b)))
print("Empty transactions excluded:", len(raw_df) - len(transactions))

print("\nFirst five transactions:")
for i, basket in enumerate(transactions[:5], start=1):
    print(f"Transaction {i}: {basket}")

# 3. Basket Size Summary & Top Products Analysis
basket_sizes = pd.Series(
    [len(basket) for basket in transactions],
    name="Basket Size"
)

print("\nAverage basket size:", round(basket_sizes.mean(), 2))
print("Median basket size:", basket_sizes.median())
print("Minimum basket size:", basket_sizes.min())
print("Maximum basket size:", basket_sizes.max())
print("\nBasket size summary:")
print(basket_sizes.describe().to_frame().to_string())

product_counts = Counter(
    item for basket in transactions for item in basket
)
top_products = pd.Series(product_counts).sort_values(
    ascending=False
).head(15)

print("\nTop 15 frequently purchased products:")
print(top_products.rename("Frequency").to_frame().to_string())

# Visualizations: Top Products & Basket Distribution
plt.figure(figsize=(12, 6))
top_products.sort_values().plot(kind="barh")
plt.title("Top 15 Most Frequently Purchased Products")
plt.xlabel("Number of Occurrences")
plt.ylabel("Product")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(basket_sizes, bins=range(1, int(basket_sizes.max()) + 2),
         edgecolor="black", align="left")
plt.title("Distribution of Products per Transaction")
plt.xlabel("Number of Products in a Transaction")
plt.ylabel("Number of Transactions")
plt.xticks(range(1, int(basket_sizes.max()) + 1))
plt.tight_layout()
plt.show()

# 4. Feature Matrix Generation (MultiLabelBinarizer)
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(transactions)
X_df = pd.DataFrame(X, columns=mlb.classes_)

print("\nFeature matrix shape:", X.shape)
print("Rows = transactions; columns = unique products")
print(X_df.head().to_string())

# 5. Principal Component Analysis (PCA)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)
explained = pca.explained_variance_ratio_

print("\nVariance explained by PC1:", round(explained[0] * 100, 2), "%")
print("Variance explained by PC2:", round(explained[1] * 100, 2), "%")
print("Combined variance explained:", round(explained.sum() * 100, 2), "%")

plt.figure(figsize=(9, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10, alpha=0.4)
plt.title("PCA Visualization of Retail Transactions")
plt.xlabel(f"Principal Component 1 ({explained[0]*100:.1f}% variance)")
plt.ylabel(f"Principal Component 2 ({explained[1]*100:.1f}% variance)")
plt.tight_layout()
plt.show()

# Sampling for Clustering Efficiency
n_sample = min(1000, len(transactions))
rng = np.random.RandomState(RANDOM_STATE)
sample_indices = np.sort(
    rng.choice(len(transactions), size=n_sample, replace=False)
)

X_sample = X[sample_indices]
X_pca_sample = X_pca[sample_indices]

print("\nTransactions used for clustering:", X_sample.shape[0])
print("Product features used:", X_sample.shape[1])

# 6. Clustering Implementations
k = 5

# --- K-Means ---
kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans.fit_predict(X_sample)
print("\nK-Means cluster sizes:")
print(pd.Series(kmeans_labels).value_counts().sort_index().to_string())

plt.figure(figsize=(9, 6))
scatter = plt.scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], c=kmeans_labels, cmap="tab10", s=18, alpha=0.7)
plt.title("K-Means Clustering of Retail Transactions (PCA View)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()

# --- Hierarchical Clustering ---
hierarchical = AgglomerativeClustering(n_clusters=k, linkage="ward")
hierarchical_labels = hierarchical.fit_predict(X_sample)
print("\nHierarchical cluster sizes:")
print(pd.Series(hierarchical_labels).value_counts().sort_index().to_string())

plt.figure(figsize=(9, 6))
scatter = plt.scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], c=hierarchical_labels, cmap="tab10", s=18, alpha=0.7)
plt.title("Hierarchical Clustering of Retail Transactions (PCA View)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()

# --- DBSCAN ---
dbscan = DBSCAN(eps=2.5, min_samples=5, metric="euclidean", n_jobs=-1)
dbscan_labels = dbscan.fit_predict(X_sample)

n_dbscan_clusters = len(set(dbscan_labels) - {-1})
n_noise = int(np.sum(dbscan_labels == -1))

print("\nDBSCAN clusters found:", n_dbscan_clusters)
print("Transactions labelled as noise:", n_noise)
print("Noise percentage:", round(n_noise / len(dbscan_labels) * 100, 2), "%")
print("\nCluster sizes (including noise label -1):")
print(pd.Series(dbscan_labels).value_counts().sort_index().to_string())

plt.figure(figsize=(9, 6))
noise_mask = dbscan_labels == -1
plt.scatter(X_pca_sample[noise_mask, 0], X_pca_sample[noise_mask, 1], c="gray", s=18, alpha=0.5, label="Noise")
cluster_mask = ~noise_mask
if cluster_mask.any():
    plt.scatter(X_pca_sample[cluster_mask, 0], X_pca_sample[cluster_mask, 1], c=dbscan_labels[cluster_mask], cmap="tab10", s=22, alpha=0.8, label="Clusters")
plt.title("DBSCAN Clustering of Retail Transactions (PCA View)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.show()

# --- Spectral Clustering ---
spectral = SpectralClustering(
    n_clusters=k,
    affinity="nearest_neighbors",
    n_neighbors=10,
    assign_labels="kmeans",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
spectral_labels = spectral.fit_predict(X_sample)
print("\nSpectral cluster sizes:")
print(pd.Series(spectral_labels).value_counts().sort_index().to_string())

plt.figure(figsize=(9, 6))
scatter = plt.scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], c=spectral_labels, cmap="tab10", s=18, alpha=0.7)
plt.title("Spectral Clustering of Retail Transactions (PCA View)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()

# 7. Evaluation Function (Safely Handles < 2 Clusters for DBSCAN)
def evaluate_clustering(X_data, labels, exclude_noise=False):
    labels = np.asarray(labels)
    mask = labels != -1 if exclude_noise else np.ones(len(labels), dtype=bool)

    X_eval = X_data[mask]
    labels_eval = labels[mask]

    unique_clusters = np.unique(labels_eval)
    n_clusters = len(unique_clusters)

    result = {
        "Number of Clusters": n_clusters,
        "Noise Points": int(np.sum(labels == -1)) if exclude_noise else 0,
        "Silhouette Coefficient": np.nan,
        "Davies-Bouldin Index": np.nan
    }

    # Safely compute metrics only if valid cluster count is >= 2
    if 2 <= n_clusters < len(labels_eval):
        result["Silhouette Coefficient"] = silhouette_score(X_eval, labels_eval)
        result["Davies-Bouldin Index"] = davies_bouldin_score(X_eval, labels_eval)

    return result

results = {
    "K-Means": evaluate_clustering(X_sample, kmeans_labels),
    "Hierarchical": evaluate_clustering(X_sample, hierarchical_labels),
    "DBSCAN": evaluate_clustering(X_sample, dbscan_labels, exclude_noise=True),
    "Spectral": evaluate_clustering(X_sample, spectral_labels)
}

results_df = pd.DataFrame(results).T
results_df.index.name = "Algorithm"

print("\nClustering performance comparison:")
print(results_df.round(4).to_string())

# Plot valid Metrics
valid_silhouette = results_df["Silhouette Coefficient"].dropna()
if not valid_silhouette.empty:
    plt.figure(figsize=(9, 5))
    valid_silhouette.sort_values().plot(kind="bar")
    plt.title("Clustering Algorithm Comparison: Silhouette Coefficient")
    plt.xlabel("Clustering Algorithm")
    plt.ylabel("Silhouette Coefficient (higher is better)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()

valid_db = results_df["Davies-Bouldin Index"].dropna()
if not valid_db.empty:
    plt.figure(figsize=(9, 5))
    valid_db.sort_values(ascending=False).plot(kind="bar")
    plt.title("Clustering Algorithm Comparison: Davies–Bouldin Index")
    plt.xlabel("Clustering Algorithm")
    plt.ylabel("Davies–Bouldin Index (lower is better)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()

# 8. Cluster Product Profiling
clustered_transactions = pd.DataFrame({
    "Transaction_Index": sample_indices,
    "KMeans_Cluster": kmeans_labels
})

cluster_product_rows = []
for cluster_id in sorted(np.unique(kmeans_labels)):
    selected_indices = sample_indices[kmeans_labels == cluster_id]
    counts = Counter(
        item
        for idx in selected_indices
        for item in transactions[idx]
    )
    for product, count in counts.most_common(10):
        cluster_product_rows.append({
            "Cluster": int(cluster_id),
            "Product": product,
            "Frequency": count
        })

cluster_products_df = pd.DataFrame(cluster_product_rows)

for cluster_id in sorted(np.unique(kmeans_labels)):
    print(f"\nTop products in K-Means Cluster {cluster_id}:")
    print(
        cluster_products_df[
            cluster_products_df["Cluster"] == cluster_id
        ][["Product", "Frequency"]].head(10).to_string(index=False)
    )

# 9. Save Outputs to CSV Files
results_df.to_csv("clustering_model_comparison.csv")
clustered_transactions.to_csv("kmeans_transaction_clusters.csv", index=False)
cluster_products_df.to_csv("kmeans_cluster_top_products.csv", index=False)

print("\nFiles saved successfully:")
print("- clustering_model_comparison.csv")
print("- kmeans_transaction_clusters.csv")
print("- kmeans_cluster_top_products.csv")