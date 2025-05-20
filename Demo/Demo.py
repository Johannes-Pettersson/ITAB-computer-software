import sys
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import librosa.display
from matplotlib.widgets import Button
import threading
from Gate_Light_Control import gate_blink_sequence, gate_sequence

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)  # Add the path to the root folder of the git repository
from FeatureSelection.GetFeatureValue import get_feature_value  # type: ignore
from System.GetFiles import get_files_from_single_dir # type: ignore
from System.FeatureExtraction import FeatureExtraction # type: ignore
from System.ZScore import ZScore, plot_z_score # type: ignore
from System.LOF import calc_lof, calc_and_plot_lof # type: ignore
from Recording.usb_mic import UsbRecorder # type: ignore

def system_prediction(input_data: FeatureExtraction, training_data: FeatureExtraction):
    z_score = ZScore()
    evaluation_values = np.ndarray((2, 1))

    size_training = len(training_data.features[training_data.feature_list[0]])
    size_features = len(training_data.feature_list)
    training_arr = np.ndarray((2, size_training))

    prediction = True # i.e non outlier unless proven otherwise
    z_score_predictions = []
    lof_predictions = []

    outlier_vs_nonoutlier_prediction_th = 0.30

    for i in range(0, size_features, 2):  # Loops through all features in pairs
        training_arr[0] = training_data.features[training_data.feature_list[i]]
        training_arr[1] = training_data.features[training_data.feature_list[i + 1]]
        z_score.train(training_arr)
        evaluation_values[0] = input_data.features[input_data.feature_list[i]]
        evaluation_values[1] = input_data.features[input_data.feature_list[i + 1]]
        z_score_predictions.append(z_score.predict(evaluation_values))
        lof_predictions.append(calc_lof(training_arr.T, evaluation_values.T))
    outlier_count = 0
    for prediction in z_score_predictions + lof_predictions:
        if not prediction:
            outlier_count+=1
    if outlier_count/len(z_score_predictions+lof_predictions) > outlier_vs_nonoutlier_prediction_th:
        prediction = False # i.e outlier

    if prediction:
        gate_blink_sequence("green")
    else:
        gate_blink_sequence("red")

def compare_waveform_with_training(fig, axs, input_file):

    assert os.path.exists(input_file), "Input file does not exist"
    inp_y, inp_sr = librosa.load(input_file)

    train_y, train_sr = librosa.load("Training_Files/G_G_5.WAV")

    axs[0].clear()
    axs[1].clear()

    axs[0].set_title("Training Audio Waveform")
    librosa.display.waveshow(train_y, sr=train_sr, ax=axs[0])

    axs[1].set_title("Input Audio Waveform")
    librosa.display.waveshow(inp_y, sr=inp_sr, ax=axs[1])

    for ax in axs:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

    fig.canvas.draw_idle()

def feature_label(feature_list):
    feature_labels = {
        "rmse_max": "Root Mean Square (max)",
        "ae_mean": "Amplitude Envelope (mean)",
        "ae_std": "Amplitude Envelope (std)",
        "rmse_std": "Root Mean Square (std)",
        "zcr_mean": "Zero Crossing Rate (mean)",
        "rmse_mean": "Root Mean Square (mean)",
        "sb_min": "Spectral Bandwidth (min)",
        "mfcc_kurtosis": "MFCC (kurtosis)",
        "sb_ptp": "Spectral Bandwidth (peak to peak)",
        "zcr_total": "Zero Crossing Rate (total)",
        "sc_min": "Spectral Centroid (min)",
        "ber_mean": "Bandwidth Energy Ratio (mean)",
        "sb_mean": "Spectral Bandwidth (mean)",
        "sc_ptp": "Spectral Centroid (peak to peak)",
    }
    return [feature_labels[feature] for feature in feature_list]

def plot_at_index(fig, axs, input_data: FeatureExtraction, training_data: FeatureExtraction, ix: int, z_score_predictions: list, lof_predictions: list):

    z_score = ZScore()
    evaluation_values = np.ndarray((2, 1))

    size_training = len(training_data.features[training_data.feature_list[0]])
    size_features = len(training_data.feature_list)
    training_arr = np.ndarray((2, size_training))

    training_arr[0] = training_data.features[training_data.feature_list[ix]]
    training_arr[1] = training_data.features[training_data.feature_list[ix + 1]]
    evaluation_values[0] = input_data.features[input_data.feature_list[ix]]
    evaluation_values[1] = input_data.features[input_data.feature_list[ix + 1]]
    z_score.train(training_arr)
    z_score_predictions[ix // 2] = z_score.predict(evaluation_values)
    lof_predictions[ix // 2] = calc_lof(training_arr.T, evaluation_values.T)

    axs[0].clear()
    axs[1].clear()
    feature_x, feature_y = feature_label(training_data.feature_list[ix : ix + 2])
    fig.suptitle(f"Prediction ( {(ix // 2) + 1} / {(size_features // 2)})")

    plot_z_score(axs[0], training_arr, evaluation_values, feature_y, feature_x, z_score_predictions[ix // 2])
    calc_and_plot_lof(axs[1], training_arr.T, evaluation_values.T, y_label = feature_y, x_label = feature_x, prediction=lof_predictions[ix // 2])

    fig.canvas.draw_idle()

def display_output(fig, axs, file_prediction):
    try:
        if file_prediction[0]:
            img_path = "sigma_gate_happy.png"
        else:
            img_path = "sigma_gate_sad.png"

        img = mpimg.imread(img_path)
        axs.imshow(img)
    except FileNotFoundError:
        print("Image file not found")
    axs.axis("off")
    title = "FINAL OUTPUT: " + ("OK" if file_prediction[0] else "ANOMALY")
    fig.suptitle(title)
    fig.canvas.draw_idle()

def state_machine_system_graphics(input_file, input_data: FeatureExtraction, training_data: FeatureExtraction):

    # ----------------- VARIABLES DECLARED START HERE ----------------

    state = ["s_waveform"]
    feature_index = [0]

    z_score_predictions = [0] * (len(input_data.feature_list) // 2)
    lof_predictions = [0] * (len(input_data.feature_list) // 2)
    file_prediction = [True]  # i.e non outlier unless proven otherwise
    outlier_vs_nonoutlier_prediction_th = 0.30

    fig, axs = plt.subplots(2, 1)
    fig.canvas.manager.full_screen_toggle()

    ax_next, ax_back, ax_close = None, None, None
    button_next, button_back, button_close = None, None, None

    # ----------------- VARIABLES DECLARED END HERE ----------------

    # ----------------- INTERNAL FUNCTIONS START HERE ----------------

    def deactivate_button(button):
        button.set_active(False)
        button.ax.set_visible(False)

    def activate_button(button):
        button.set_active(True)
        button.ax.set_visible(True)

    def on_next(event):

        if state[0] == "s_waveform":
            state[0] = "s_features"
            recreate_figure(1, 2)
            feature_index[0] = 0
            plot_at_index(fig, axs, input_data, training_data, feature_index[0], z_score_predictions, lof_predictions)

        elif state[0] == "s_features":
            if feature_index[0] < len(input_data.feature_list) - 2:
                feature_index[0] += 2
                plot_at_index(fig, axs, input_data, training_data, feature_index[0], z_score_predictions, lof_predictions)
                if feature_index[0] == len(input_data.feature_list) - 2:
                    outlier_count = 0
                    for prediction in z_score_predictions + lof_predictions:
                        if not prediction:
                            outlier_count += 1
                    if outlier_count / len(z_score_predictions + lof_predictions) > outlier_vs_nonoutlier_prediction_th:
                        file_prediction[0] = False

            elif feature_index[0] >= len(input_data.feature_list) - 2:
                state[0] = "s_result"
                recreate_figure(1, 1)
                display_output(fig, axs, file_prediction)

        elif state[0] == "s_result":
            pass


    def on_back(event):
        if state[0] == "s_waveform":
            pass

        elif state[0] == "s_features":
            if feature_index[0] >= 2:
                feature_index[0] -= 2
                plot_at_index(fig, axs, input_data, training_data, feature_index[0], z_score_predictions, lof_predictions)
            else:
                state[0] = "s_waveform"
                recreate_figure(2, 1)
                compare_waveform_with_training(fig, axs, input_file)
        elif state[0] == "s_result":
            state[0] = "s_features"
            recreate_figure(1, 2)
            feature_index[0] = len(input_data.feature_list) - 2
            plot_at_index(fig, axs, input_data, training_data, feature_index[0], z_score_predictions, lof_predictions)

    def on_close(event):
        plt.close()

    def recreate_figure(nrows, ncols):
        nonlocal axs, ax_next, ax_back, ax_close
        nonlocal button_next, button_back, button_close

        fig.clf()

        axs = fig.subplots(nrows, ncols)
        fig.subplots_adjust(bottom=0.2)

        button_width = 0.12
        button_height = 0.06
        button_y = 0.02

        back_x = 0.25 - button_width / 2
        next_x = 0.5 - button_width / 2
        close_x = 0.75 - button_width / 2

        ax_back = plt.axes([back_x, button_y, button_width, button_height])
        button_back = Button(ax_back, "Back")
        button_back.on_clicked(on_back)
        activate_button(button_back)

        ax_next = plt.axes([next_x, button_y, button_width, button_height])
        button_next = Button(ax_next, "Next")
        button_next.on_clicked(on_next)
        activate_button(button_next)

        ax_close = plt.axes([close_x, button_y, button_width, button_height])
        button_close = Button(ax_close, "Close")
        button_close.on_clicked(on_close)
        activate_button(button_close)

    # ----------------- INTERNAL FUNCTIONS END HERE ----------------

    recreate_figure(2, 1)
    compare_waveform_with_training(fig, axs, input_file)
    plt.show()

def retrain_system():
    try:
        os.remove("training_data.pkl")
        print("Pickle file deleted")
    except FileNotFoundError:
        print("No pickle file found for deletion")

    usb_recorder = UsbRecorder()
    for i in range(50):
        file_name = f"Training_Files/G_G_{i}.WAV"
        gate_sequence_th = threading.Thread(target=gate_sequence, daemon=True)
        gate_sequence_th.start()
        usb_recorder.record(file_name)

        gate_sequence_th.join()

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

    file_name = "Input_Files/Input.WAV"
    input_file = [file_name]

    gate_sequence_th = threading.Thread(target=gate_sequence, daemon=True)
    gate_sequence_th.start()

    usb_recorder = UsbRecorder()
    usb_recorder.record(file_name)

    gate_sequence_th.join()

    input_data = FeatureExtraction(feature_list, input_file)
    training_data = None

    try:
        with open("training_data.pkl", "rb") as f:
            training_data = pickle.load(f)
    except FileNotFoundError:
        print("No training data found, creating new one...")
        num_of_training_files = 50
        training_files = get_files_from_single_dir(num_of_training_files, "Training_Files/")
        training_data = FeatureExtraction(feature_list, training_files)
        with open("training_data.pkl", "wb") as f:
            pickle.dump(training_data, f)

    prediction_th = threading.Thread(target=system_prediction, args=(input_data, training_data), daemon=True)
    prediction_th.start()

    state_machine_system_graphics(input_file[0], input_data, training_data)
    prediction_th.join()

if __name__ == "__main__":
    main()
