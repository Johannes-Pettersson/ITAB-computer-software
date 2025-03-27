import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt


# Generate random datar
data_set = np.random.randn(100,2)
#test_point = np.array([15,12])

estimator = LocalOutlierFactor(n_neighbors=20)

full_data = np.vstack([data_set]) #,test_point])
labels = estimator.fit_predict(full_data)


X_scores = estimator.negative_outlier_factor_
outlier_colors = [(1,0,0,0.5) if label == -1 else (0,0,0,0) for label in labels]

plt.scatter(data_set[:, 0], data_set[:, 1], color="k", s=3.0, label="Data points")


# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
scatter = plt.scatter(
    data_set[:, 0],
    data_set[:, 1],
    s=1000 * radius,
    edgecolors="r",
    facecolors=outlier_colors,
    label="Outlier scores",
)
plt.axis("tight")
plt.title("Local Outlier Factor (LOF)")
plt.show()