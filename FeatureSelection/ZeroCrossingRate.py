import librosa
import numpy as np

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
    # Parameters
    frame_size = 1024
    hop_size = 512
    y, sr = librosa.load(file)
    zcr_total_val = sum(librosa.zero_crossings(y))
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_size, hop_length=hop_size)[0]
    t = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=hop_size)
    zcr_mean = np.mean(zcr)
    zcr_max = np.max(zcr)
    zcr_std = np.std(zcr)
    return y, sr, t, zcr_total_val, zcr, zcr_mean, zcr_max, zcr_std

def main():
    import matplotlib.pyplot as plt
    titles = [
        "../Recording/Functioning_gate_recordings/Day 2/Session 1/G_G_1.WAV",
        "../Recording/Functioning_gate_recordings/Day 2/Session 2/G_G_1.WAV",
        "../Recording/Faulty_gate_recordings/Day 2/Session 1/B_G_1.WAV",
        "../Recording/Faulty_gate_recordings/Day 2/Session 2/B_G_1.WAV",
    ]
    audio_data = [calculate_values(file) for file in titles]
    fig, axes = plt.subplots(2, 2)
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
