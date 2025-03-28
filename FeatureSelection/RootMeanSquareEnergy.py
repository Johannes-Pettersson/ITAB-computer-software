import librosa
import numpy as np

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
    frame_size = 1024
    hop_size = 512
    y, sr = librosa.load(file)
    rms = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_size)[0]
    frames = range(0, rms.size)
    t = librosa.frames_to_time(frames, hop_length=hop_size)
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
    import matplotlib.pyplot as plt
    titles = [
        "../Recording/Functioning_gate_recordings/Day 2/Session 1/G_G_1.WAV",
        "../Recording/Functioning_gate_recordings/Day 2/Session 2/G_G_1.WAV",
        "../Recording/Faulty_gate_recordings/Day 2/Session 1/B_G_1.WAV",
        "../Recording/Faulty_gate_recordings/Day 2/Session 2/B_G_1.WAV",
    ]
    audio_data = [calculate_values(file) for file in titles]
    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 2)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

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
