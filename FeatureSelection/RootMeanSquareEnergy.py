import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa
import random
import os

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
    Calculate the mean, max and std values of the RMSE of an audio file.
    Uses librosa to load the audio file and compute the RMSE,
    with FRAME_SIZE and HOP_SIZE as parameters (1024 and 512 by default).

    Parameters
    ----------
    file : str Path to the audio file.

    Returns
    -------
        y : audio time series. Multi-channel is supported (librosa.load).

        sr : sampling rate of y (librosa.load).

        t : time (in seconds) of each given frame number (librosa.frames_to_time).

        rms : RMS value for each frame (librosa.feature.rms).

        mean_val : Mean value of the RMSE (numpy.mean).

        max_val : Maximum value of the RMSE (numpy.max).

        std_val : Standard deviation of the RMSE (numpy.std).
    """
    y, sr = librosa.load(file)
    rms = librosa.feature.rms(y=y, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)[0]
    frames = range(0, rms.size)
    t = librosa.frames_to_time(frames, hop_length=HOP_SIZE)
    mean_val = np.mean(rms)
    max_val = np.max(rms)
    std_val = np.std(rms)
    return (
        y,
        sr,
        t,
        rms,
        mean_val,
        max_val,
        std_val,
    )


def main():
    # Process audio files
    n_good_files = 2
    n_faulty_files = 2
    titles = get_files(n_good_files, n_faulty_files)
    audio_data = [calculate_values(file) for file in titles]
    # Set up the figure with subplots
    fig, axes = plt.subplots(n_good_files, n_faulty_files, figsize=(15, 7.25))

    # Plot each waveform and amplitude envelope
    for ax, (y, sr, t, rms, mean_val, max_val, std_val), title in zip(
        axes.flatten(), audio_data, titles
    ):
        ax.set_title(
            f"{title}\nMean: {mean_val:.4f}, Max: {max_val:.4f}, Std: {std_val:.4f}"
        )
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.plot(t, rms, color="r")

    # Set consistent axis limits
    max_time = max(t[-1] for _, _, t, _, _, _, _ in audio_data)
    for ax in axes.flatten():
        ax.set_xlim([0, max_time])
        ax.set_ylim([-1, 1])  # Assuming normalized audio data

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
