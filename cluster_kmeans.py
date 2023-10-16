import pandas as pd
from sklearn.cluster import KMeans
import pickle

# Load data
data = pd.read_csv('processed_data_clustering.csv')

# excluding 'unitid' and 'year'
data_clustering = data.drop(columns=['unitid', 'year'])

# K-Means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
data['cluster'] = kmeans.fit_predict(data_clustering)

# Save model
with open('kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)

# Save cluster assignments
data.to_csv('kmeans_clustered_data.csv', index=False)
