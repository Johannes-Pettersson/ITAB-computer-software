from LOF import calc_lof
from GetFiles import get_files
from ZScore import ZScore
from FeatureExtraction import FeatureExtraction
import sys
import os
import numpy as np
import pickle

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)  # Add the path to the root folder of the git repository
from FeatureSelection.GetFeatureValue import get_feature_value  # type: ignore

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

    return "OK" if file_prediction else "ANOMALY"

def compare_waveform_with_training(input_file):
    import matplotlib.pyplot as plt
    import librosa.display

    inp_y, inp_sr = librosa.load(input_file)

    train_y, train_sr = librosa.load("Exjobb/Training_Files/G_G_5.WAV")

    fig, axs = plt.subplots(2, 1)
    fig.canvas.manager.full_screen_toggle()

    axs[0].set_title("Training Audio Waveform")
    librosa.display.waveshow(train_y, sr=train_sr, ax=axs[0])

    axs[1].set_title("Input Audio Waveform")
    librosa.display.waveshow(inp_y, sr=inp_sr, ax=axs[1])

    for ax in axs:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

    plt.tight_layout()
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
    training_data = None
    input_file = get_files(1, "Exjobb/Input_Files/")
    input_data = FeatureExtraction(feature_list, input_file)
    assert len(input_file) == 1, "Only one input file is allowed"
    print(f"Predicting for files: {input_file}...")
    compare_waveform_with_training(input_file[0])

    try:
        with open("Exjobb/training_data.pkl", "rb") as f:
            training_data = pickle.load(f)
    except FileNotFoundError:
        print("No training data found, creating new one...")
        num_of_training_files = 50
        training_files = get_files(num_of_training_files, "Exjobb/Training_Files/")
        training_data = FeatureExtraction(feature_list, training_files)
        with open("Exjobb/training_data.pkl", "wb") as f:
            pickle.dump(training_data, f)

    prediction = anomaly_system_prediction(training_data=training_data, input_data=input_data)
    print(f"OUTPUT: {prediction}")

if __name__ == "__main__":
    main()
