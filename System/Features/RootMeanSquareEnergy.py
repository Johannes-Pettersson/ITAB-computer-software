import numpy as np
import librosa

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

if __name__ == "__main__":
    print("Dont run this file solo")
