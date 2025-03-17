import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Parameters
FRAME_SIZE = 1024
HOP_SIZE = 512


def get_files(num_of_good_files, num_of_faulty_files):
    good_files = []
    faulty_files = []

    good_directories = [
        "../Recording/Functioning_gate_recordings/Day 2/Session 1",
        "../Recording/Functioning_gate_recordings/Day 2/Session 2",
    ]
    faulty_directories = [
        "../Recording/Faulty_gate_recordings/Day 2/Session 1",
        "../Recording/Faulty_gate_recordings/Day 2/Session 2",
    ]

    for dir in good_directories:
        for entry in os.scandir(dir):
            if entry.is_file():
                good_files.append(entry.path)

    for dir in faulty_directories:
        for entry in os.scandir(dir):
            if entry.is_file():
                faulty_files.append(entry.path)

    while len(good_files) > num_of_good_files:
        good_files.pop(random.randrange(len(good_files)))

    while len(faulty_files) > num_of_faulty_files:
        faulty_files.pop(random.randrange(len(faulty_files)))

    return good_files + faulty_files


def calculate_values(file):
    """
    Calculate the mean, max and std values of the Zero Crossing Rate of an audio file.
    Uses librosa to load the audio file and compute the Zero Crossing Rate,
    with FRAME_SIZE and HOP_SIZE as parameters (1024 and 512 by default).
    
    Parameters
    ----------
    file : str Path to the audio file.

    Returns
    -------
        y : audio time series. Multi-channel is supported (librosa.load).

        sr : sampling rate of y (librosa.load).

        t : time of each frame (librosa.frames_to_time).

        zcr_total_val : total number of zero crossings in the audio file.

        zcr : zero crossing rate of each frame.

        zcr_mean : mean value of the zero crossing rate.

        zcr_max : maximum value of the zero crossing rate.

        zcr_std : standard deviation of the zero crossing
    """
    y, sr = librosa.load(file)
    zcr_total_val = sum(librosa.zero_crossings(y))
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)[0]
    t = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=HOP_SIZE)
    zcr_mean = np.mean(zcr)
    zcr_max = np.max(zcr)
    zcr_std = np.std(zcr)
    return y, sr, t, zcr_total_val, zcr, zcr_mean, zcr_max, zcr_std



def main():
    n_good_files = 2
    n_faulty_files = 2
    titles = get_files(n_good_files, n_faulty_files)
    audio_data = [calculate_values(file) for file in titles]
    fig, axes = plt.subplots(n_good_files, n_faulty_files, figsize=(80, 60))
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    for ax, (y, sr, t, zcr_total_val, zcr, zcr_mean, zcr_max, zcr_std), title in zip(
        axes.flatten(), audio_data, titles
    ):
        ax.set_title(
            f"{title}\nTotal: {zcr_total_val:.4f}, Mean: {zcr_mean:.4f}, Max: {zcr_max:.4f}, Std: {zcr_std:.4f}"
        )
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.plot(t, zcr, color="r")

    max_time = max(t[-1] for _, _, t, _, _, _, _, _ in audio_data)
    for ax in axes.flatten():
        ax.set_xlim([0, max_time])
        ax.set_ylim([-1, 1])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
