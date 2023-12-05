import pickle

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Load data
data = pd.read_csv('processed_data_clustering.csv')

# exclude 'unitid' and 'year'
data_clustering = data.drop(columns=['unitid', 'year'])

# K-Means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
data['cluster'] = kmeans.fit_predict(data_clustering)

# Train KNN model for each cluster
knn_models = {}
for cluster in range(10):
    # Filter data for the current cluster
    cluster_data = data[data['cluster'] == cluster]

    # Store original indices and unitid
    original_indices = cluster_data.index
    unitids = cluster_data['unitid']

    # Drop 'cluster', 'unitid', and 'year' columns for KNN training
    cluster_data = cluster_data.drop(columns=['cluster', 'unitid', 'year'])

    # Train KNN model
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(cluster_data)
    knn_models[cluster] = (knn, original_indices, unitids)  # Store the model, original indices, and unitids

# save trained K-Means and KNN
with open('kmeans_knn_model.pkl', 'wb') as file:
    pickle.dump((kmeans, knn_models), file)

# Save cluster assignments
data.to_csv('kmeans_clustered_data.csv', index=False)
