import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from math import ceil

def calc_lof_outlier_factor(data: np.ndarray, data_point, n_neighbors=20):
    # Number of neighbors is either 20 or 50% of the length of the training data.
    n_neighbors = min(n_neighbors, ceil(len(data)*0.5))

    estimator = LocalOutlierFactor(n_neighbors=n_neighbors)
    labels = estimator.fit_predict(np.vstack([data, data_point]))

    if labels is None:
        raise Exception("Labels didn't get a value")

    x_scores = estimator.negative_outlier_factor_

    outlier_factors = (x_scores.max() - x_scores) / (x_scores.max() - x_scores.min())


    return outlier_factors[-1]



def calc_lof(data: np.ndarray, data_point, n_neighbors=20) -> bool:
    "Calculate if the data_point is a outlier. False == Outlier, True == Non-Outlier"

    # Number of neighbors is either 20 or 50% of the length of the training data.
    n_neighbors = min(n_neighbors, ceil(len(data)*0.5))

    estimator = LocalOutlierFactor(n_neighbors=n_neighbors)

    labels = estimator.fit_predict(np.vstack([data, data_point]))

    if labels is None:
        raise Exception("Labels didn't get a value")

    return labels[-1] == 1


def calc_and_plot_lof(data: np.ndarray, data_point=None, n_neighbors=20):
    # Number of neighbors is either 20 or 50% of the length of the training data.
    n_neighbors = min(n_neighbors, ceil(len(data)*0.5))

    estimator = LocalOutlierFactor(n_neighbors=n_neighbors)

    labels = None
    if data_point is not None:
        data = np.vstack([data, data_point])

    labels = estimator.fit_predict(data)

    if labels is None:
        raise Exception("Labels didn't get a value")

    X_scores = estimator.negative_outlier_factor_

    if data.shape[1] == 2:
        if data_point is not None:
            dot_colors = ["k" if i < len(labels)-1 else "g" for i in range(len(labels))]
        else:
            dot_colors = ["k" for _ in range(len(labels))]
        plt.scatter(data[:, 0], data[:, 1], color=dot_colors, s=3.0)

        perimiter_colors = [(1,0,0,0.5) if label == -1 else (0,0,0,0) for label in labels]
        if perimiter_colors[-1] != (0,0,0,0):
            perimiter_colors[-1] = (0,1,0,0.5)

        radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
        plt.scatter(
            data[:, 0],
            data[:, 1],
            s=1000 * radius,
            edgecolors=["r" if color == "k" else "g" for color in dot_colors],
            facecolors=perimiter_colors,
            label="Outlier scores",
        )
    elif data.shape[1] == 1:
        if data_point is not None:
            dot_colors = ["k" if i < len(labels)-1 else "g" for i in range(len(labels))]
        else:
            dot_colors = ["k" for _ in range(len(labels))]
        plt.scatter(data, np.zeros(len(data)), color=dot_colors, s=3.0)

        perimiter_colors = [(1,0,0,0.5) if label == -1 else (0,0,0,0) for label in labels]
        if perimiter_colors[-1] != (0,0,0,0):
            perimiter_colors[-1] = (0,1,0,0.5)

        radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
        plt.scatter(
            data,
            np.zeros(len(data)),
            s=1000 * radius,
            edgecolors=["r" if color == "k" else "g" for color in dot_colors],
            facecolors=perimiter_colors,
            label="Outlier scores",
        )

    plt.axis("tight")
    if data_point is not None:
        plt.title(f"LOF. Outlier? -> {labels[-1] == -1}")
    else:
        plt.title("LOF")
    plt.show()

def main():

    data_set_size = 10
    outlier_values = []


    for i in range(1000):
        data_set = np.random.randn(data_set_size,2)
        factor = calc_lof_outlier_factor(data_set[:-1], data_set[-1])
        if not calc_lof(data_set[:-1], data_set[-1]):
            outlier_values.append(factor)

    print(f"The average outlier value for size {data_set_size} is {np.mean(outlier_values)}")
    print(f"Min: {min(outlier_values)}")
    print(f"Max: {max(outlier_values)}")



if __name__ == "__main__":
    main()
