import librosa
import matplotlib.pyplot as plt
import numpy as np

def band_energy_ratio(spectrogram, split_frequency, sample_rate):
    """Calculate band energy ratio with a given split frequency."""
    
    split_frequency_bin = calculate_split_frequency_bin(split_frequency, sample_rate, len(spectrogram[0]))
    band_energy_ratio = []
    
    # calculate power spectrogram
    power_spectrogram = np.abs(spectrogram) ** 2
    power_spectrogram = power_spectrogram.T
    
    # calculate BER value for each frame
    for frame in power_spectrogram:
        sum_power_low_frequencies = frame[:split_frequency_bin].sum()
        sum_power_high_frequencies = frame[split_frequency_bin:].sum()
        band_energy_ratio_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        band_energy_ratio.append(band_energy_ratio_current_frame)
    
    return np.array(band_energy_ratio)

def calculate_values(file):
    y, sr = librosa.load(file)
    S, phase = librosa.magphase(librosa.stft(y))
    
    b_e_ratio = band_energy_ratio(S, 1000, sr)

    print(f"Band energy ratio: {b_e_ratio}")

    return y, sr, S, b_e_ratio


def calculate_split_frequency_bin(split_frequency, sample_rate, num_frequency_bins):
    """Infer the frequency bin associated to a given split frequency."""
    
    frequency_range = sample_rate / 2
    frequency_delta_per_bin = frequency_range / num_frequency_bins
    split_frequency_bin = np.floor(split_frequency / frequency_delta_per_bin)
    return int(split_frequency_bin)

def plot_file(file, col, axes, hop_length):

    y, sr, S, b_e_ratio = calculate_values(file)

    S_db = librosa.amplitude_to_db(S, ref=np.max)

    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, 
                                   x_axis="time", y_axis="log", ax=axes[0, col])
    title = file.split("/")[-2] + "/" + file.split("/")[-1]
    axes[0, col].set_title(title) 

    times = librosa.times_like(b_e_ratio, sr=sr, hop_length=hop_length)

    axes[1, col].plot(times, b_e_ratio, label="Band Energy Ratio")
    axes[1, col].set_xlabel("Time (s)")
    axes[1, col].set_ylabel("BER")

def main():

    
    FRAME_SIZE = 2048
    HOP_SIZE = 512

    faulty_files = [
    "../Recording/Faulty_gate_recordings/Day 1/Session 1/B_G_1.WAV"
    ]

    functioning_files = [
        "../Recording/Functioning_gate_recordings/Day 2/Session 2/G_G_176.WAV"
    ]

    fig, axes = plt.subplots(nrows=2, ncols=len(faulty_files)+ len(functioning_files), figsize=(12, 8), sharex=True, sharey=False)

    for i, file in enumerate(faulty_files + functioning_files):
        plot_file(file, i, axes=axes, hop_length=HOP_SIZE)


    # Sync axels of b_e_ratio plots
    y_min = np.inf
    y_max = -np.inf

    for ax in axes[1, :]:
        lines = ax.get_lines()
        for line in lines:
            y_data = line.get_ydata()
            y_min = min(y_min, np.min(y_data))
            y_max = max(y_max, np.max(y_data))

    for ax in axes[1, :]:
        ax.set_ylim(y_min, y_max)

    plt.show()

if __name__ == "__main__":
    main()