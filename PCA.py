import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

clustered_data = ['kmeans_clustered_data.csv', 'kmodes_clustered_data.csv']

for data_file in clustered_data:
    # Load clustered data
    data = pd.read_csv(data_file)

    # remove 'unitid', 'year', and 'cluster'
    data_pca = data.drop(columns=['unitid', 'year', 'cluster'])

    # PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data_pca)

    # DataFrame pca
    pca_df = pd.DataFrame(data=pca_data, columns=['Component 1', 'Component 2'])
    pca_df['Cluster'] = data['cluster']

    # Plot
    plt.figure(figsize=(12, 8))
    for cluster in pca_df['Cluster'].unique():
        subset = pca_df[pca_df['Cluster'] == cluster]
        plt.scatter(subset['Component 1'], subset['Component 2'], label=f'Cluster {cluster}', s=50)

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'Clusters Visualized in 2D PCA Space {data_file}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plot/Clusters Visualized in 2D PCA Space {data_file}.png')
    plt.show()
