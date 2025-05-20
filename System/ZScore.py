import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # Add the path to the root folder of the git repository
from FeatureSelection.RootMeanSquareEnergy import calculate_values as rmse_calculate_values  # type: ignore
from FeatureSelection.ZeroCrossingRate import calculate_values as zcr_calculate_values  # type: ignore
from FeatureSelection.MelFrequencyCepstralCoefficients import calculate_values as mfcc_calculate_values  # type: ignore


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

def plot_z_score(ax, training_data: np.ndarray, data_point: np.ndarray, y_label, x_label, prediction):
    """Plots the training data and the data point with the z-score radius."""

    if training_data.ndim != 2:
        raise ValueError(
            f"The training data must be a 2D array, data input is {training_data.ndim}D"
        )

    if data_point.ndim != 2:
        raise ValueError(
            f"The input data must be a 2D array, data input is {data_point.ndim}D"
        )
    mean_x = np.mean(training_data[0])
    mean_y = np.mean(training_data[1])
    std_dev_x = np.std(training_data[0])
    std_dev_y = np.std(training_data[1])
    z_score_training_x = (training_data[0] - mean_x) / std_dev_x
    z_score_training_y = (training_data[1] - mean_y) / std_dev_y

    z_score_other_x = (data_point[0] - mean_x) / std_dev_x
    z_score_other_y = (data_point[1] - mean_y) / std_dev_y

    title = "Z-Score - Predicts: " + ("OK" if prediction else "ANOMALY")
    ax.set_title(title)
    ax.scatter(z_score_training_x, z_score_training_y, color="green", s=20.0, label="Training Data")
    ax.scatter(z_score_other_x, z_score_other_y, color="red", s=20.0, label="Input Data")

    ax.fill_between(x=np.linspace(-3, 3, 100), y1=-3, y2=3, color="green", alpha=0.1)
    ax.legend()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def main():
    from GetFiles import get_files
    num_of_good_gate_files = 20
    num_of_faulty_gate_files = 10
    num_of_training_files = 10
    good_gate_files, faulty_gate_files = get_files(
        num_of_good_gate_files, num_of_faulty_gate_files
    )

    train_data = np.ndarray((2, num_of_training_files))
    input_data = np.ndarray((2, num_of_faulty_gate_files))
    another_input_data = np.ndarray((2, num_of_good_gate_files - num_of_training_files))

    for i in range(num_of_training_files):
        file = good_gate_files[i]
        _, _, _, _, rmse_mean_val, rmse_max_val, rmse_std_val = rmse_calculate_values(
            file
        )
        _, _, _, _, _, zcr_mean_val, zcr_max_val, zcr_std_val = zcr_calculate_values(
            file
        )
        _, _, _, mfcc_skewness, _ = mfcc_calculate_values(file, coef=1, dct_type=4)
        train_data[0, i] = mfcc_skewness
        train_data[1, i] = zcr_mean_val

    for i in range(num_of_faulty_gate_files):
        file = faulty_gate_files[i]
        _, _, _, _, rmse_mean_val, rmse_max_val, rmse_std_val = rmse_calculate_values(
            file
        )
        _, _, _, _, _, zcr_mean_val, zcr_max_val, zcr_std_val = zcr_calculate_values(
            file
        )
        _, _, _, mfcc_skewness, _ = mfcc_calculate_values(file, coef=1, dct_type=4)
        input_data[0, i] = mfcc_skewness
        input_data[1, i] = zcr_mean_val

    for i in range(num_of_good_gate_files - num_of_training_files):
        file = good_gate_files[i + num_of_training_files]
        _, _, _, _, rmse_mean_val, rmse_max_val, rmse_std_val = rmse_calculate_values(
            file
        )
        _, _, _, _, _, zcr_mean_val, zcr_max_val, zcr_std_val = zcr_calculate_values(
            file
        )
        _, _, _, mfcc_skewness, _ = mfcc_calculate_values(file, coef=1, dct_type=4)
        another_input_data[0, i] = mfcc_skewness
        another_input_data[1, i] = zcr_mean_val

    plot_z_score(train_data, input_data)
    z_score = ZScore()
    z_score.train(train_data)

    accuracy = 0
    val = np.ndarray((2, 1))

    # Prediction for faulty gates. Expected that the prediction to be False, since the z_score is trained on good gates.
    for i in range(num_of_faulty_gate_files):
        val[0] = input_data[0, i]
        val[1] = input_data[1, i]
        if not z_score.predict(val):
            accuracy += 1
        else:
            print("")
            print("z_score faulty gates")
            print(val)
            print("^^^False Positive^^^")
            print("")

    # Prediction for good gates. Expected that the prediction to be True, since the z_score is trained on good gates.
    for i in range(num_of_good_gate_files - num_of_training_files):
        val[0] = another_input_data[0, i]
        val[1] = another_input_data[1, i]
        if z_score.predict(val):
            accuracy += 1
        else:
            print("")
            print("z_score good gates")
            print(val)
            print("^^^False Negative^^^")
            print("")

    accuracy = (
        accuracy
        / (num_of_good_gate_files - num_of_training_files + num_of_faulty_gate_files)
    ) * 100
    print(
        f"Accuracy: {accuracy}%, for {num_of_good_gate_files-num_of_training_files} good gates and {num_of_faulty_gate_files} faulty gates"
    )


if __name__ == "__main__":
    main()
