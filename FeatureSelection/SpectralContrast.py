import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

faulty_files = [
    "../Recording/Faulty_gate_recordings/Session 1/B_G_100.WAV"
]

functioning_files = [
    "../Recording/Functioning_gate_recordings/Session 2/G_G_23.WAV"
]

fig, axes = plt.subplots(nrows=2, ncols=len(faulty_files) + len(functioning_files), figsize=(12, 8), sharex=True)
roll_percent = .37

def plot_file(file, col, axes, fig):
    y, sr = librosa.load(file)

    S = np.abs(librosa.stft(y))

    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    img1 = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=axes[0, col])
    fig.colorbar(img1, ax=[axes[0, col]], format='%+2.0f dB')
    title = file.split("/")[-2] + "/" + file.split("/")[-1]
    axes[0, col].set_title(f"{title}\nPower spectogram")
    axes[0, col].label_outer()

    img2 = librosa.display.specshow(contrast, x_axis='time', ax=axes[1, col])
    fig.colorbar(img2, ax=axes[1, col])
    axes[1, col].set(ylabel="Frequency bands", title="Spectral contrast")


for i, file in enumerate(faulty_files + functioning_files):
    plot_file(file, i, axes=axes, fig=fig)



#y, sr = librosa.load(functioning_files[1])

#sd.play(y, sr)
#sd.wait()


plt.show()
