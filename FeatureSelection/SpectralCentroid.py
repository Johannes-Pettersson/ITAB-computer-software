import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

faulty_files = [
    "../Recording/Faulty_gate_recordings/Session 1/B_G_1.WAV",
    "../Recording/Faulty_gate_recordings/Session 1/B_G_77.WAV"
]

functioning_files = [
    "../Recording/Functioning_gate_recordings/Session 1 (some ticking)/G_G_1.WAV",
    "../Recording/Functioning_gate_recordings/Session 1 (some ticking)/G_G_33.WAV"
]

fig, axes = plt.subplots(nrows=2, ncols=max(len(faulty_files), len(functioning_files)), figsize=(12, 8), sharex=True, sharey=True)

for i, file in enumerate(faulty_files):
    y, sr = librosa.load(file)
    S, phase = librosa.magphase(librosa.stft(y))
    cent = librosa.feature.spectral_centroid(S=S)
    times = librosa.times_like(cent)

    ax = axes[0, i] 
    
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
    ax.plot(times, cent.T, label="Spectral centroid", color='w')
    ax.legend(loc='upper right')
    title = file.split("/")[-2] + "/" + file.split("/")[-1]
    ax.set_title(f"{title}")  

    if i != 0:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

for i, file in enumerate(functioning_files):
    y, sr = librosa.load(file)
    S, phase = librosa.magphase(librosa.stft(y))
    cent = librosa.feature.spectral_centroid(S=S)
    times = librosa.times_like(cent)

    ax = axes[1, i] 
    
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
    ax.plot(times, cent.T, label="Spectral centroid", color='w')
    ax.legend(loc='upper right')
    title = file.split("/")[-2] + "/" + file.split("/")[-1]
    ax.set_title(f"{title}")  

    if i != 0:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

y, sr = librosa.load(functioning_files[1])

sd.play(y, sr)
sd.wait()


plt.tight_layout()
plt.show()
