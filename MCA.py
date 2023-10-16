import pandas as pd
import matplotlib.pyplot as plt
import prince  # This is the library for MCA

clustered_data = ['kmeans_clustered_data.csv', 'kmodes_clustered_data.csv']

for data_file in clustered_data:
    # Load data
    data = pd.read_csv(data_file)

    # remove 'unitid', 'year', and 'cluster'
    data_mca = data.drop(columns=['unitid', 'year', 'cluster'])

    # MCA
    mca = prince.MCA(n_components=2)
    mca_data = mca.fit_transform(data_mca)

    # Plot
    plt.figure(figsize=(12, 8))
    for cluster in data['cluster'].unique():
        subset = mca_data[data['cluster'] == cluster]
        plt.scatter(subset[0], subset[1], label=f'Cluster {cluster}', s=50)

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'Clusters Visualized in 2D MCA Space {data_file}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plot/Clusters Visualized in 2D MCA Space {data_file}.png')
    plt.show()
