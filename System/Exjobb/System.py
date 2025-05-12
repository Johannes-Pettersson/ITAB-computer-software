from LOF import calc_lof
from GetFiles import get_files
from ZScore import ZScore
from FeatureExtraction import FeatureExtraction
import sys
import os
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
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


def anomaly_system_prediction(
    training_data: FeatureExtraction,
    input_data: FeatureExtraction,
):

    z_score = ZScore()
    evaluation_values = np.ndarray((2, 1))

    size_training = len(training_data.features[training_data.feature_list[0]])
    size_features = len(training_data.feature_list)

    outlier_vs_nonoutlier_prediction_th = 0.30

    training_arr = np.ndarray((2, size_training))

    file_prediction = True # i.e non outlier unless proven otherwise
    z_score_predictions = []
    lof_predictions = []
    for k in range(0, size_features, 2):  # Loops through all feature values in list
        training_arr[0] = training_data.features[training_data.feature_list[k]]
        training_arr[1] = training_data.features[training_data.feature_list[k + 1]]
        z_score.train(training_arr)
        evaluation_values[0] = input_data.features[input_data.feature_list[k]]
        evaluation_values[1] = input_data.features[input_data.feature_list[k + 1]]
        z_score_predictions.append(z_score.predict(evaluation_values))
        lof_predictions.append(calc_lof(training_arr.T, evaluation_values.T))
    outlier_count = 0
    for prediction in z_score_predictions + lof_predictions:
        if not prediction:
            outlier_count+=1
    if outlier_count/len(z_score_predictions+lof_predictions) > outlier_vs_nonoutlier_prediction_th:
        file_prediction = False # i.e outlier

    return file_prediction

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

    num_good_gate_files = 50 + 1
    # Get files for training data
    files = get_files(num_good_gate_files)
    # Extract the feature data for training
    training_data = FeatureExtraction(feature_list, files[ : -1])
    # Extract the feature data for input data
    input_data = FeatureExtraction(feature_list, files[-1 : ])
    print(f"Predicting for files: {files[-1 : ]}")
    prediction = anomaly_system_prediction(training_data=training_data, input_data=input_data)
    print(f"Prediction for file: {files[-1 : ]} is: {prediction}")

if __name__ == "__main__":
    main()
