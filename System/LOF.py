import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt


def calc_lof(data: np.ndarray, data_point, n_neighbors=20) -> bool:
    "Calculate if the data_point is a outlier. False == Outlier, True == Non-Outlier"
    estimator = LocalOutlierFactor(n_neighbors=n_neighbors)

    labels = estimator.fit_predict(np.vstack([data, data_point]))
    return labels[-1] == 1


def calc_and_plot_lof(data: np.ndarray, data_point=None, n_neighbors=20):
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

    data_set = np.random.randn(100,2)
    calc_and_plot_lof(data_set[:-1], data_set[-1])


if __name__ == "__main__":
    main()
