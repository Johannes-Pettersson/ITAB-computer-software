import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Function to compute amplitude envelope
def amplitude_envelope(signal, frame_size, hop_size):
    amplitude_envelope = []
    for i in range(0, len(signal), hop_size):
        current_frame_amplitude_envelope = max(signal[i:i+frame_size])
        amplitude_envelope.append(current_frame_amplitude_envelope)
    return np.array(amplitude_envelope)


def calculate_values(file_path):
    """
    Calculate the mean, max and std values of the Amplitude Envelope of an audio file.
    Uses librosa to load the audio file and compute the Amplitude Envelope,
    with FRAME_SIZE and HOP_SIZE as parameters (1024 and 512 by default).
    
    Parameters
    ----------
        file_path : str Path to the audio file.
        
    Returns
    -------
        y : audio time series. Multi-channel is supported (librosa.load).
        
        sr : sampling rate of y (librosa.load).
        
        ae : amplitude envelope of the audio file.
        
        t : time of each frame (librosa.frames_to_time).
        
        mean_val : mean value of the amplitude envelope.
        
        max_val : maximum value of the amplitude envelope.
        
        std_val : standard deviation of the amplitude envelope.
    """
    frame_size = 1024
    hop_size = 512
    y, sr = librosa.load(file_path)
    ae = amplitude_envelope(y, frame_size, hop_size)
    t = librosa.frames_to_time(range(len(ae)), hop_length=hop_size)
    mean_val = np.mean(ae)
    max_val = np.max(ae)
    std_val = np.std(ae)
    return y, sr, ae, t, mean_val, max_val, std_val


def main():
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

    for ax, (y, sr, ae, t, mean_val, max_val, std_val), title in zip(axes.flatten(), audio_data, titles):
        ax.set_title(f"{title}\nMean: {mean_val:.4f}, Max: {max_val:.4f}, Std: {std_val:.4f}")
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.plot(t, ae, color='r')

    max_time = max(t[-1] for _, _, _, t, _, _, _ in audio_data)
    for ax in axes.flatten():
        ax.set_xlim([0, max_time])
        ax.set_ylim([-1, 1])
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()