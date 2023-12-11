import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

clustered_data = ['kmeans_clustered_data.csv']

scaler = MinMaxScaler()

for data_file in clustered_data:
    # Load clustered data
    data = pd.read_csv(data_file)

    # Remove 'unitid', 'year', and 'cluster'
    data_tsne = data.drop(columns=['unitid', 'cluster'])

    # Standardize the data
    data_tsne_scaled = scaler.fit_transform(data_tsne)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    tsne_data = tsne.fit_transform(data_tsne_scaled)

    # DataFrame for t-SNE
    tsne_df = pd.DataFrame(data=tsne_data, columns=['Dimension 1', 'Dimension 2'])
    tsne_df['Cluster'] = data['cluster']

    # Plot
    plt.figure(figsize=(12, 8))
    for cluster in tsne_df['Cluster'].unique():
        subset = tsne_df[tsne_df['Cluster'] == cluster]
        plt.scatter(subset['Dimension 1'], subset['Dimension 2'], label=f'Cluster {cluster}', s=50)

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'Clusters Visualized in 2D t-SNE Space {data_file}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plot/Clusters Visualized in 2D t-SNE Space {data_file}.png')
    plt.show()
