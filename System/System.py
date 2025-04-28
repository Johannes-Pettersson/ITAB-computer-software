from LOF import calc_lof, calc_and_plot_lof
from GetFiles import get_files, get_files_from_directories
from ZScore import ZScore, plot_z_score
from FeatureExtraction import FeatureExtraction
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # Add the path to the root folder of the git repository
from FeatureSelection.GetFeatureValue import get_feature_value  # type: ignore


def get_feature_value_list(
    good_gate_files, faulty_gate_files, feature_1_type, feature_2_type
):
    functioning_feature_1_values = []
    faulty_feature_1_values = []

    functioning_feature_2_values = []
    faulty_feature_2_values = []

    for file in good_gate_files:
        functioning_feature_1_values.append(get_feature_value(feature_1_type, file))
        functioning_feature_2_values.append(get_feature_value(feature_2_type, file))

    for file in faulty_gate_files:
        faulty_feature_1_values.append(get_feature_value(feature_1_type, file))
        faulty_feature_2_values.append(get_feature_value(feature_2_type, file))

    combined_good_gate_feature_values = np.vstack(
        (functioning_feature_1_values, functioning_feature_2_values)
    ).T

    combined_faulty_gate_feature_values = np.vstack(
        (faulty_feature_1_values, faulty_feature_2_values)
    ).T

    return combined_good_gate_feature_values, combined_faulty_gate_feature_values


def anomaly_detection_evaluation(
    training_data: FeatureExtraction,
    evaluation_data: FeatureExtraction,
    expected_results,
):

    z_score = ZScore()
    evaluation_values = np.ndarray((2, 1))

    size_training = len(training_data.features[training_data.feature_list[0]])
    size_evaluation = len(evaluation_data.features[evaluation_data.feature_list[0]])
    size_features = len(training_data.feature_list)

    outlier_vs_nonoutlier_prediction_th = 0.30

    accuracies = []

    for i in range(1, size_training):  # Loops through Quantity of training data (2 -> 50)

        accuracy = 0
        training_arr = np.ndarray((2, i + 1))
        for j in range(size_evaluation):  # Loops through all evaluation datapoints (1 -> 100)

            total_file_prediction = True
            z_score_predictions = []
            lof_predictions = []
            for k in range(0, size_features, 2):  # Loops through all features in list

                training_arr[0] = training_data.features[training_data.feature_list[k]][: i + 1]

                training_arr[1] = training_data.features[training_data.feature_list[k + 1]][: i + 1]

                z_score.train(training_arr)

                evaluation_values[0] = evaluation_data.features[evaluation_data.feature_list[k]][j]
                evaluation_values[1] = evaluation_data.features[evaluation_data.feature_list[k + 1]][j]

                z_score_predictions.append(z_score.predict(evaluation_values))
                lof_predictions.append(calc_lof(training_arr.T, evaluation_values.T))

            outlier_count = 0
            for prediction in z_score_predictions + lof_predictions:
                if not prediction:
                    outlier_count+=1

            if outlier_count/len(z_score_predictions+lof_predictions) > outlier_vs_nonoutlier_prediction_th:
                total_file_prediction = False # i.e outlier

            if total_file_prediction == expected_results[j]:
                accuracy += 1

        accuracy = (accuracy / size_evaluation) * 100
        accuracies.append(accuracy)

    return accuracies


def plot_accuracy(accuracies, labels):
    plt.figure(figsize=(8, 5))
    colors = ["b", "r", "g"]
    for accuracy, label, color in zip(accuracies, labels, colors):
        x_values = range(2, len(accuracy) + 2)
        plt.plot(
            x_values,
            accuracy,
            linestyle="-",
            linewidth=3.0,
            color=color,
            label=label,
        )

    plt.xlabel("Quantity of Training Data (Files)")
    plt.ylabel("Accuracy (%)")
    plt.title("Performance Metrics")

    plt.xlim(0, 50)
    plt.ylim(0, 100)
    plt.xticks(range(0, 51, 5))
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def main():
    # Insert what features to use here
    feature_list = [
        "rmse_max",
        "ae_mean",
        "ae_std",
        "rmse_std",
        "zcr_mean",
        "rmse_mean",
        "sb_min",
        "mfcc_kurtosis",
        "sb_ptp",
        "zcr_total",
        "sc_min",
        "ber_mean",
        "sb_mean",
        "sc_ptp",
    ]
    # Set paths to the dataset directories
    dataset_directories = [
        "../Dataset/Training/G_G_F_0",
        "../Dataset/Training/F_G_F_0",
        "../Dataset/Training/C_G_F_0",
        "../Dataset/Evaluation/G_G_F_0",
        "../Dataset/Evaluation/F_G_F_0",
    ]
    labels = ["Good Gates", "Faulty Gates", "Combined"]
    accuracies = []

    num_good_gate_files = 50
    num_faulty_gate_files = 50
    num_combined_files = 25
    # Get files for training
    good_gate_files, faulty_gate_files = get_files_from_directories(
        num_good_gate_files,
        num_faulty_gate_files,
        good_gate_dir=dataset_directories[0],
        faulty_gate_dir=dataset_directories[1],
        pick_randomly=False,
    )
    # combined_good_gate_files, combined_faulty_gate_files = get_files_from_directories(
    #     num_combined_files,
    #     num_combined_files,
    #     good_gate_dir=dataset_directories[2],
    #     faulty_gate_dir=dataset_directories[2],
    #     pick_randomly=False
    # )

    # Extract the feature data for training
    training_good_gate_features = FeatureExtraction(feature_list, good_gate_files)
    training_faulty_gate_features = FeatureExtraction(feature_list, faulty_gate_files)
    # training_combined_good_gate_features = FeatureExtraction(feature_list, combined_good_gate_files)
    # training_combined_faulty_gate_features = FeatureExtraction(feature_list, combined_faulty_gate_files)

    # Get files for evaluation
    good_gate_files, faulty_gate_files = get_files_from_directories(
        num_good_gate_files,
        num_faulty_gate_files,
        good_gate_dir=dataset_directories[3],
        faulty_gate_dir=dataset_directories[4],
        pick_randomly=False,
    )

    # # Extract the feature data for evaluation
    evaluation_features = FeatureExtraction(
        feature_list, good_gate_files + faulty_gate_files
    )
    accuracies.append(
        anomaly_detection_evaluation(
            training_good_gate_features,
            evaluation_features,
            [True] * len(good_gate_files) + [False] * len(faulty_gate_files),
        )
    )

    accuracies.append(
        anomaly_detection_evaluation(
            training_faulty_gate_features,
            evaluation_features,
            [False] * len(good_gate_files) + [True] * len(faulty_gate_files),
        )
    )
    # Fake data remove when real data is available
    acc = []
    for i in range(50):
        acc.append(15 + i)

    accuracies.append(acc)

    plot_accuracy(accuracies, labels)
    # good_gate_feature_values, faulty_gate_feature_values = get_feature_value_list(
    #     good_gate_files=good_gate_files,
    #     faulty_gate_files=faulty_gate_files,
    #     feature_1_type="sc_min",
    #     feature_2_type="sb_max")

    # calc_and_plot_lof(faulty_gate_feature_values[:-1], faulty_gate_feature_values[-1])


if __name__ == "__main__":
    main()
