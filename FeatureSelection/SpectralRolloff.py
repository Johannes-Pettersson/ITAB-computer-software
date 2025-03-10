import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

faulty_files = [
    "../Recording/Faulty_gate_recordings/Session 1/B_G_3.WAV",
    "../Recording/Faulty_gate_recordings/Session 1/B_G_56.WAV"
]

functioning_files = [
    "../Recording/Functioning_gate_recordings/Session 2/G_G_15.WAV",
    "../Recording/Functioning_gate_recordings/Session 2/G_G_124.WAV"
]

fig, axes = plt.subplots(nrows=2, ncols=max(len(faulty_files), len(functioning_files)), figsize=(12, 8), sharex=True, sharey=True)
roll_percent = .37

def plot_file(file, row, col, axes):
    y, sr = librosa.load(file)

    S, phase = librosa.magphase(librosa.stft(y))

    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=roll_percent)

    times = librosa.times_like(rolloff)

    ax = axes[row, col] 
    
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
    ax.plot(times, rolloff[0], label=f"Roll-off frequency ({roll_percent})", color='w')
    ax.legend(loc='upper right')
    title = file.split("/")[-2] + "/" + file.split("/")[-1]
    ax.set_title(f"{title}")  

    if col != 0:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

for i, file in enumerate(faulty_files):
    plot_file(file, 0, i, axes=axes)

for i, file in enumerate(functioning_files):
    plot_file(file, 1, i, axes=axes)


#y, sr = librosa.load(functioning_files[1])

#sd.play(y, sr)
#sd.wait()


plt.tight_layout()
plt.show()
