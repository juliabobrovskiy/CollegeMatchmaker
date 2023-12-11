import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load data
data = pd.read_csv('processed_data_clustering.csv')

# Exclude 'unitid' and 'year'
data_clustering = data.drop(columns=['unitid'])

# Standardize the data
scaler = MinMaxScaler()
data_clustering_scaled = scaler.fit_transform(data_clustering)

# K-Means clustering
kmeans = KMeans(n_clusters=14, random_state=42)
data['cluster'] = kmeans.fit_predict(data_clustering_scaled)

# Train KNN model for each cluster
knn_models = {}
for cluster in range(10):
    # Filter data for the current cluster
    cluster_data = data[data['cluster'] == cluster]

    # Store original indices and unitid
    original_indices = cluster_data.index
    unitids = cluster_data['unitid']

    # Drop 'cluster' and 'unitid' columns for KNN training
    cluster_data = cluster_data.drop(columns=['cluster', 'unitid'])

    # Standardize cluster data
    cluster_data_scaled = scaler.transform(cluster_data)

    # Train KNN model
    knn = NearestNeighbors(n_neighbors=6)
    knn.fit(cluster_data_scaled)
    knn_models[cluster] = (knn, original_indices, unitids)  # Store the model, original indices, and unitids

# Save trained K-Means and KNN
with open('kmeans_knn_model.pkl', 'wb') as file:
    pickle.dump((kmeans, knn_models), file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Save cluster assignments
data.to_csv('kmeans_clustered_data.csv', index=False)
