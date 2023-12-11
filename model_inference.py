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

# New students' data
new_student_data = [
    [3.07692308e+00, 3.68461538e+01, 0.00000000e+00, 1.00000000e+00,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     3.61538462e+00, 8.46153846e-01, 1.00000000e+00, 1.00000000e+00,
     4.54545455e-01, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     6.36363636e-01, 4.54545455e-01, 0.00000000e+00, 5.45454545e-01,
     5.00000000e-01, 7.07908425e+00, 0.00000000e+00, 8.18181818e-01,
     7.10412028e-01, 1.18199695e+04],
    [9.00000000e+00, 1.11111111e+01, 1.00000000e+00, 2.85714286e+00,
     1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
     9.44444444e-01, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
     6.11111111e-01, 1.00000000e+00, 1.00000000e+00, 9.44444444e-01,
     1.00000000e+00, 7.07908425e+00, 5.55555556e-01, 4.44444444e-01,
     5.91822732e-01, 7.88060185e+03],
    [9.00000000e+00, 1.10000000e+01, 1.00000000e+00, 4.00000000e+00,
     0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     2.00000000e+00, 2.00000000e+00, 6.00000000e+00, 3.00000000e+00,
     1.00000000e+00, 2.00000000e+00, 1.00000000e+00, 1.11111111e-01,
     1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
     0.00000000e+00, 1.26000000e+01, 0.00000000e+00, 0.00000000e+00,
     7.45603299e-02, 3.84754722e+04]
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

    print(f"Recommendations for student {idx + 1}:\n")
    print(recommended_indices)
    print("\n" + "-" * 50 + "\n")
