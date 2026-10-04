import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment

data_path = r"C:\\Users\\dthan\\Downloads\\Iris.csv" 
data = pd.read_csv(data_path)

X = data.iloc[:, :-1].values  
y_true = data.iloc[:, -1].values  
label_encoder = LabelEncoder()
y_true_encoded = label_encoder.fit_transform(y_true)

def kmeans(X, k, epsilon=1e-5, max_iterations=300):
    n_samples, n_features = X.shape
    centers = X[np.random.choice(n_samples, k, replace=False)]
    prev_centers = np.zeros_like(centers)
    labels = np.zeros(n_samples)

    for iteration in range(max_iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)

        prev_centers = centers.copy()
        for i in range(k):
            points = X[labels == i]
            if len(points) > 0:
                centers[i] = np.mean(points, axis=0)

        if np.linalg.norm(centers - prev_centers) < epsilon:
            break

    return labels, centers

k = 3
predicted_labels, final_centers = kmeans(X, k)

conf_matrix = confusion_matrix(y_true_encoded, predicted_labels)
row_ind, col_ind = linear_sum_assignment(-conf_matrix)
optimal_labels = np.zeros_like(predicted_labels)
for true_label, pred_label in zip(row_ind, col_ind):
    optimal_labels[predicted_labels == pred_label] = true_label

success_rate = accuracy_score(y_true_encoded, optimal_labels)

print("Confusion Matrix:")
print(conf_matrix)
print(f"Success Rate: {success_rate:.4f}")