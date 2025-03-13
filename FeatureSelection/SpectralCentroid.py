import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.signal import savgol_filter

faulty_files = [
    "../Recording/Faulty_gate_recordings/Session 1/B_G_1.WAV"
]

functioning_files = [
    "../Recording/Functioning_gate_recordings/Session 1 (some ticking)/G_G_4.WAV"
]

fig, axes = plt.subplots(nrows=2, ncols=len(faulty_files)+ len(functioning_files), figsize=(12, 8), sharex=True, sharey=False)

def plot_file(file, row, col, axes):
    y, sr = librosa.load(file)
    S, phase = librosa.magphase(librosa.stft(y))
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)
    times = librosa.times_like(cent)

    ax = axes[row, col] 
    
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
    ax.plot(times, cent.T, label="Spectral centroid", color='w')
    ax.legend(loc='upper right')

    min_val = cent.min()
    max_val = cent.max()
    ptp_val = np.ptp(cent)

    title = file.split("/")[-2] + "/" + file.split("/")[-1]
    ax.set_title(f"{title}\nMin: {min_val:.0f}, Max: {max_val:.0f} ptp: {ptp_val:.0f}")  

    if col != 0:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    # Plotta derivatan -----------------
    ax_deriv = axes[row+1, col]

    window_size = 7

    smoothed_dx_dt = savgol_filter(cent.T.flatten(), window_size, polyorder=2, deriv=1, delta=np.mean(np.diff(times)))
    ax_deriv.plot(times, smoothed_dx_dt, label="Derivata (Savgol)", color="r")

    ax_deriv.legend(loc='upper right')
    ax_deriv.grid(True, linestyle="--", alpha=0.5)
    ax_deriv.set_ylabel("Förändring i centroid")

    ax_deriv.set_title(f"Max: {max(smoothed_dx_dt):.0f}, Min: {min(smoothed_dx_dt):.0f}")
    # ------------------------


for i, file in enumerate(faulty_files + functioning_files):
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

# y, sr = librosa.load(faulty_files[0])

# sd.play(y, sr)
# sd.wait()


plt.tight_layout()
plt.show()
