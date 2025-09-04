from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Dataset_Day14.csv")
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use DBSCAN  (e.g., eps=0.8, min_samples=4)
dbscan = DBSCAN(eps=0.8, min_samples=4)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Identify outliers (label -1)
outliers = df[df['DBSCAN_Cluster'] == -1]

# Output statements
print("\n Outlier Detection using DBSCAN")
print(f"Total number of points in dataset: {len(df)}")
print(f" Total outliers detected (cluster label -1): {len(outliers)}")
print("\nOutlier details:")
print(outliers[['Id', 'Species', 'DBSCAN_Cluster']])


