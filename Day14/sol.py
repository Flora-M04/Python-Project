import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset and check for missing values
df = pd.read_csv("Dataset_Day14.csv")
print("\n Missing values in each column:\n")
print(df.isnull().sum())

# Drop missing rows if any (optional)
df.dropna(inplace=True)
print(f"\nData shape after removing missing values: {df.shape}")

# Extract numeric features
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[features]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n  Features have been scaled using StandardScaler.")

# 2. Fit DBSCAN with default values (eps=0.8, min_samples=4)
print("\n: Applying DBSCAN with eps=0.8, min_samples=4 ...")
dbscan_default = DBSCAN(eps=0.8, min_samples=4)
df['Cluster_default'] = dbscan_default.fit_predict(X_scaled)

# Show species distribution in clusters
print("\n Species distribution in default DBSCAN clusters:")
print(pd.crosstab(df['Cluster_default'], df['Species']))

# 3. Use Nearest Neighbors to find optimal eps
print("\n  Plotting k-distance graph to estimate optimal eps")
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
k_distance = np.sort(distances[:, 4])

# Plot k-distance graph
plt.figure(figsize=(8, 4))
plt.plot(k_distance)
plt.title("K-distance Graph to Determine Optimal eps")
plt.xlabel("Data Points Sorted")
plt.ylabel("5th Nearest Neighbor Distance")
plt.grid(True)
plt.show()

# 4. Use eps=0.8 and find optimal min_samples
print("\n Silhouette scores for min_samples from 3 to 9 (eps=0.8):")
eps_val = 0.8
best_score = -1
best_min_samples = 0

for min_samples in range(3, 10):
    model = DBSCAN(eps=eps_val, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    if len(set(labels)) > 1 and -1 in labels and len(set(labels)) > 2:
        score = silhouette_score(X_scaled, labels)
        print(f"min_samples = {min_samples}, Silhouette Score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_min_samples = min_samples

if best_score > 0:
    print(f"\n Best min_samples = {best_min_samples} with silhouette score = {best_score:.4f}")
else:
    print("\nNo suitable silhouette score found for any min_samples value.")

#  Find and show outliers (label -1)
outliers = df[df['Cluster_default'] == -1]
print(f"\n  Total outliers found: {outliers.shape[0]}")
print("\n Outlier entries:")
print(outliers[['Id', 'Species', 'Cluster_default']])
