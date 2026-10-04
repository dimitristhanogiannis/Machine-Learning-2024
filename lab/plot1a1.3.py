import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

priors = np.array([0.2, 0.5, 0.3])
means = np.array([[3, -2], [4, 3], [6, 7]])
cov = np.array([[1, 0.5], [0.5, 1]]) 

n_samples = 500
samples_class1 = np.random.multivariate_normal(means[0], cov, n_samples)
samples_class2 = np.random.multivariate_normal(means[1], cov, n_samples)
samples_class3 = np.random.multivariate_normal(means[2], cov, n_samples)

plt.figure(figsize=(10, 8))
plt.scatter(samples_class1[:, 0], samples_class1[:, 1], color='blue', alpha=0.5, label="Κλάση ω1")
plt.scatter(samples_class2[:, 0], samples_class2[:, 1], color='green', alpha=0.5, label="Κλάση ω2")
plt.scatter(samples_class3[:, 0], samples_class3[:, 1], color='red', alpha=0.5, label="Κλάση ω3")

plt.scatter(means[0][0], means[0][1], color='blue', edgecolor='black', marker='x', s=100, label="Κέντρο Κλάσης ω1")
plt.scatter(means[1][0], means[1][1], color='green', edgecolor='black', marker='x', s=100, label="Κέντρο Κλάσης ω2")
plt.scatter(means[2][0], means[2][1], color='red', edgecolor='black', marker='x', s=100, label="Κέντρο Κλάσης ω3")

x_vals = np.linspace(-1, 10, 200)
y_vals = np.linspace(-1, 10, 200)
X, Y = np.meshgrid(x_vals, y_vals)
pos = np.dstack((X, Y))

likelihoods_class1 = multivariate_normal.pdf(pos, mean=means[0], cov=cov)
likelihoods_class2 = multivariate_normal.pdf(pos, mean=means[1], cov=cov)
likelihoods_class3 = multivariate_normal.pdf(pos, mean=means[2], cov=cov)

post_class1 = likelihoods_class1 * priors[0]
post_class2 = likelihoods_class2 * priors[1]
post_class3 = likelihoods_class3 * priors[2]

plt.contour(X, Y, post_class1 - post_class2, levels=[0], colors='purple')
plt.contour(X, Y, post_class1 - post_class3, levels=[0], colors='orange')
plt.contour(X, Y, post_class2 - post_class3, levels=[0], colors='brown')

plt.title("Σημεία και Καμπύλες Απόφασης Κλάσεων")
plt.xlabel("X")
plt.ylabel("Υ")
plt.legend()
plt.grid(True)
plt.show()

n_mc_samples = 1000
samples_class2_mc = np.random.multivariate_normal(means[1], cov, n_mc_samples)

misclassified_count = 0

for sample in samples_class2_mc:
    likelihood_c1 = multivariate_normal.pdf(sample, mean=means[0], cov=cov) * priors[0]
    likelihood_c2 = multivariate_normal.pdf(sample, mean=means[1], cov=cov) * priors[1]
    likelihood_c3 = multivariate_normal.pdf(sample, mean=means[2], cov=cov) * priors[2]
    
    total_likelihood = likelihood_c1 + likelihood_c2 + likelihood_c3
    post_c1 = likelihood_c1 / total_likelihood
    post_c2 = likelihood_c2 / total_likelihood
    post_c3 = likelihood_c3 / total_likelihood
    
    predicted_class = np.argmax([post_c1, post_c2, post_c3]) + 1
    
    if predicted_class != 2:
        misclassified_count += 1

misclassification_prob = misclassified_count / n_mc_samples
print(misclassification_prob)
