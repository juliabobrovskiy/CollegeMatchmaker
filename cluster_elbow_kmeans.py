import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import os

# Load data
data = pd.read_csv('processed_data_clustering.csv')

# Dropping 'unitid' and 'year'
data_clustering = data.drop(columns=['unitid'])

# Standardize the data
scaler = MinMaxScaler()
data_clustering_scaled = scaler.fit_transform(data_clustering)

# Elbow method
inertia = []
K = range(1, 25)  # Up to 14 clusters
for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans.fit(data_clustering_scaled)
    inertia.append(kmeans.inertia_)

# Plot inertia against K values
plt.figure(figsize=(10, 6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.grid(True)

# Make plot directory and save the plot
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/Elbow Method For Optimal Number of Clusters.png')
plt.show()
