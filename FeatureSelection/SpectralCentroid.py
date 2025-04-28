import librosa
import matplotlib.pyplot as plt
import numpy as np
# import sounddevice as sd
from scipy.signal import savgol_filter

faulty_gate_files = [
    "../Recording/Faulty_gate_recordings/Day 1/Session 1/B_G_1.WAV"
]

good_gate_files = [
    "../Recording/Functioning_gate_recordings/Day 2/Session 2/G_G_176.WAV"
]

def calculate_values(file):
    y, sr = librosa.load(file)
    S, phase = librosa.magphase(librosa.stft(y))
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)
    times = librosa.times_like(cent)

    min_val = cent.min()
    max_val = cent.max()
    ptp_val = np.ptp(cent)

    window_size = 7

    smoothed_dx_dt = savgol_filter(cent.T.flatten(), window_size, polyorder=2, deriv=1, delta=np.mean(np.diff(times)))

    max_deriv = max(smoothed_dx_dt)
    min_deriv = min(smoothed_dx_dt)

    return S, cent, times, min_val, max_val, ptp_val, smoothed_dx_dt, max_deriv, min_deriv


def plot_file(file, row, col, axes):
    
    S, cent, times, min_val, max_val, ptp_val, smoothed_dx_dt, max_deriv, min_deriv = calculate_values(file)

    ax = axes[row, col] 
    
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
    ax.plot(times, cent.T, label="Spectral centroid", color='w')
    ax.legend(loc='upper right')

    title = file.split("/")[-2] + "/" + file.split("/")[-1]
    ax.set_title(f"{title}\nMin: {min_val:.0f}, Max: {max_val:.0f} ptp: {ptp_val:.0f}")  

    if col != 0:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    ax_deriv = axes[row+1, col]

    ax_deriv.plot(times, smoothed_dx_dt, label="Derivata (Savgol)", color="r")

    ax_deriv.legend(loc='upper right')
    ax_deriv.grid(True, linestyle="--", alpha=0.5)
    ax_deriv.set_ylabel("Förändring i centroid")

    ax_deriv.set_title(f"Max: {max_deriv:.0f}, Min: {min_deriv:.0f}")

def main():
    fig, axes = plt.subplots(nrows=2, ncols=len(faulty_gate_files)+ len(good_gate_files), figsize=(12, 8), sharex=True, sharey=False)

    for i, file in enumerate(faulty_gate_files + good_gate_files):
        plot_file(file, 0, i, axes=axes)


    # Sync axels of derivetive plots
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



    plt.tight_layout()
    plt.show()

    # y, sr = librosa.load(faulty_gate_files[0])

    # sd.play(y, sr)
    # sd.wait()

if __name__ == "__main__":
    main()