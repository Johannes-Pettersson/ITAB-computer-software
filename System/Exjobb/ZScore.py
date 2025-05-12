import numpy as np
import matplotlib.pyplot as plt


class ZScore:
    """Class for calculating the z-score of the input data."""

    def __init__(self):
        self.mean_x = None
        self.mean_y = None
        self.std_dev_x = None
        self.std_dev_y = None
        self.threshold_x = 3.0
        self.threshold_y = 3.0

    def train(self, training_data: np.ndarray):
        """Calculates and sets the mean and standard deviation of the training data."""
        if training_data.ndim != 2:
            raise ValueError(
                f"The training data must be a 2D array, data input is {training_data.ndim}D"
            )
        self.mean_x = np.mean(training_data[0])
        self.mean_y = np.mean(training_data[1])
        self.std_dev_x = np.std(training_data[0])
        self.std_dev_y = np.std(training_data[1])

    def predict(self, data_point: np.ndarray) -> bool:
        """Calculates z_score of input values, compares this value to threshold. Returns True if the z_score is within the threshold, False otherwise."""
        if self.mean_x is None:
            raise ValueError("The model has not been trained")

        if data_point.ndim != 2:
            raise ValueError(
                f"The input data must be a 2D array, data input is {data_point.ndim}D"
            )

        z_score_x = (data_point[0] - self.mean_x) / self.std_dev_x
        z_score_y = (data_point[1] - self.mean_y) / self.std_dev_y

        predict_x = abs(z_score_x) <= self.threshold_x
        predict_y = abs(z_score_y) <= self.threshold_y

        return predict_x and predict_y

    def predict_combination(self, data_point: np.ndarray) -> float:
        """Calculates z_score of input values, and returns the mean of the z_scores."""
        if self.mean_x is None:
            raise ValueError("The model has not been trained")

        if data_point.ndim != 2:
            raise ValueError(
                f"The input data must be a 2D array, data input is {data_point.ndim}D"
            )

        z_score_x = (data_point[0] - self.mean_x) / self.std_dev_x
        z_score_y = (data_point[1] - self.mean_y) / self.std_dev_y

        return (abs(z_score_x) + abs(z_score_y)) / 2.0

if __name__ == "__main__":
    pass
