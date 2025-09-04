import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Load dataset
df = pd.read_csv('Dataset_Day13.csv')

# Check for missing values
print(df.isnull().sum())

# fill or drop missing values
df.dropna(inplace=True)  

# Detect and treat outliers (using IQR method)
numeric_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
print("After cleaning, dataset shape:", df.shape)
print("\n Preview of cleaned data:")
print(df.head())

X = df[numeric_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Group by species and calculate mean petal length
avg_petal_length = df.groupby('Species')['PetalLengthCm'].mean()
print(avg_petal_length)

# Only numeric columns grouped by Species
desc_stats = df.groupby('Species')[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].describe()
print(desc_stats)

# Elbow Method 
inertias = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(K, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.show()

# Silhouette and Calinski-Harabasz Scores for odd k 
print("\nSilhouette and Calinski-Harabasz Scores:")
for k in [3, 5, 7, 9]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    print(f"k = {k} | Silhouette Score = {sil:.3f} | Calinski-Harabasz Score = {ch:.2f}")

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

proportion_table = pd.crosstab(df['Cluster'], df['Species'], normalize='index')
print("\n Proportion of Species in Each Cluster:")
print(proportion_table)

sns.pairplot(df, hue='Cluster', vars=numeric_cols, palette='Set2')
plt.suptitle("K-Means Cluster Visualization", y=1.02)
plt.show()

print("\n Insights based on Clustering:")

if proportion_table.shape[0] >= 3:
    cluster_0 = proportion_table.iloc[0].idxmax()
    cluster_1 = proportion_table.iloc[1].idxmax()
    cluster_2 = proportion_table.iloc[2].idxmax()
    print(f"- Cluster 0 mostly contains: {cluster_0}")
    print(f"- Cluster 1 mostly contains: {cluster_1}")
    print(f"- Cluster 2 mostly contains: {cluster_2}")

print("""
- The Elbow method shows a sharp drop at k = 3, supporting it as the optimal number of clusters.
- Silhouette and Calinski-Harabasz scores also peak or are relatively high at k = 3.
- Cluster 0 is almost purely Iris-setosa, indicating it is linearly separable from the others.
- Versicolor and Virginica are mixed in Cluster 1 and 2 — these species are not linearly separable.
- K-Means was effective at separating Setosa but not perfect for the other two.
""")