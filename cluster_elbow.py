import pandas as pd
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import os

# Load data
data = pd.read_csv('processed_data_clustering.csv')

# dropping 'unitid' and 'year'
data_clustering = data.drop(columns=['unitid', 'year'])

# elbow method
cost = []
K = range(1, 14)  # up to 14 clusters
for k in K:
    kmodes = KModes(n_clusters=k, init='Huang', n_init=5, verbose=1)
    kmodes.fit(data_clustering)
    cost.append(kmodes.cost_)

# Plot cost against K values
plt.figure(figsize=(10, 6))
plt.plot(K, cost, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Clustering cost')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.grid(True)

# make plot directory and save the plot
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/Elbow Method For Optimal Number of Clusters.png')
plt.show()
