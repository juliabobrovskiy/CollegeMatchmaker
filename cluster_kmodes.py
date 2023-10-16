import pandas as pd
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import os

# Load data
data = pd.read_csv('processed_data_clustering.csv')

# dropping 'unitid' and 'year'
data_clustering = data.drop(columns=['unitid', 'year'])

# K-Modes clustering
kmodes = KModes(n_clusters=10, init='Huang', n_init=5, verbose=1)
clusters = kmodes.fit_predict(data_clustering)

data['cluster'] = clusters

# Save luster assignments
data.to_csv('kmodes_clustered_data.csv', index=False)
