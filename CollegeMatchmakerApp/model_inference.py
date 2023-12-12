import pickle

import pandas as pd

# Load models
with open('kmeans_knn_model.pkl', 'rb') as file:
    kmeans_loaded, knn_models_loaded = pickle.load(file)

# Assuming you've saved the scaler as 'scaler.pkl'
with open('scaler.pkl', 'rb') as file:
    scaler_loaded = pickle.load(file)

# Load clustered data
data = pd.read_csv('kmeans_clustered_data.csv')
# load unitid mapping (to get names)
unitid_map = pd.read_csv('unitid_mapping.csv')

# New students' data
new_student_data = [
    [3, 40, 0, 1, 0, 0, 0, 0, 5, 2, 6,
     3, 0, 2, 0, 0, 1, 1, 1, 0, 0, 7,
     0, 1, 7, 8685],
    [9, 11, 1, 4, 0, 1, 0, 0, 2,
     2, 6, 3, 1, 2, 1, 0, 1, 1,
     1, 1, 0, 12, 0, 0, 0, 38475]  # <-- this is stanford
]

# Scale new student data
new_student_data_scaled = scaler_loaded.transform(new_student_data)

# Assign student to a cluster using K-Means
cluster_assignments = kmeans_loaded.predict(new_student_data_scaled)

# Find k nearest universities using KNN model for each student
for idx, cluster in enumerate(cluster_assignments):
    knn_model, original_indices, unitids = knn_models_loaded[cluster]
    distances, indices = knn_model.kneighbors([new_student_data_scaled[idx]])

    recommended_indices = unitids.iloc[indices[0][1:]].values
    # Map these unit IDs to institution names
    recommended_institutions = unitid_map[unitid_map['unitid'].isin(recommended_indices)]['inst_name'].values
    clusters = data[data['unitid'].isin(recommended_indices)]['cluster'].values

    print(f"Recommendations for student {idx + 1}:\n")
    print('Recommended unitids:', recommended_indices)
    print('Recommended inst_name:', recommended_institutions)
    print('Clusters', clusters)
    print("\n" + "-" * 50 + "\n")
