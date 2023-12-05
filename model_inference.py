import pickle
import pandas as pd

# Load models
with open('kmeans_knn_model.pkl', 'rb') as file:
    kmeans_loaded, knn_models_loaded = pickle.load(file)

# Load clustered data
data = pd.read_csv('kmeans_clustered_data.csv')

# new students' data
new_student_data = [
    [9.0, 3.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    [3.0, 4.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
]

# Assign student to a cluster using K-Means
cluster_assignments = kmeans_loaded.predict(new_student_data)

# Find k nearest universities using KNN model for each student
for idx, cluster in enumerate(cluster_assignments):
    knn_model, original_indices, unitids = knn_models_loaded[cluster]
    distances, indices = knn_model.kneighbors([new_student_data[idx]])

    recommended_indices = unitids.iloc[indices[0]].values

    print(f"Recommendations for student {idx + 1}:\n")
    print(recommended_indices)
    print("\n" + "-" * 50 + "\n")
