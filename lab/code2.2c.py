import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score

file_path = r"C:\\Users\\dthan\\Downloads\\Iris.csv"
data = pd.read_csv(file_path)

X = data[["PetalLengthCm", "PetalWidthCm"]]

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

species_to_numeric = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
data['Species_numeric'] = data['Species'].map(species_to_numeric)

cm = confusion_matrix(data['Species_numeric'], data['Cluster'])

cluster_to_species = {i: cm[:, i].argmax() for i in range(3)}

data['Mapped_Cluster'] = data['Cluster'].map(cluster_to_species)

if data['Mapped_Cluster'].isna().any():
    print("Warning: Unmapped cluster values found!")

success_rate = accuracy_score(data['Species_numeric'], data['Mapped_Cluster'])

print("Confusion Matrix:")
print(cm)
print(f"Success Rate: {success_rate:.4f}")

plt.figure(figsize=(8, 6))
for cluster, species_num in cluster_to_species.items():
    cluster_points = X[data['Cluster'] == cluster]
    species_name = list(species_to_numeric.keys())[list(species_to_numeric.values()).index(species_num)]
    plt.scatter(cluster_points["PetalLengthCm"], cluster_points["PetalWidthCm"], label=f'{species_name}')

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')

plt.title("k-means Clustering (Petal Features)")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.show()