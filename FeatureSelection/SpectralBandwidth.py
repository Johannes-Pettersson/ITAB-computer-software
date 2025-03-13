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

def plot_file(file, col, axes, fig):
    y, sr = librosa.load(file)

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    S, phase = librosa.magphase(librosa.stft(y=y))

    times = librosa.times_like(spec_bw)

    centroid = librosa.feature.spectral_centroid(S=S)


    axes[0, col].semilogy(times, spec_bw[0], label='Spectral bandwidth')
    axes[0, col].set(ylabel='Hz', xticks=[], xlim=[times.min(), times.max()])
    axes[0, col].label_outer()
    axes[0, col].legend(loc='lower right')
    title = file.split("/")[-2] + "/" + file.split("/")[-1]
    axes[0, col].set_title(f"{title}")  


    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=axes[1, col])

    axes[1, col].set(title='log Power spectrogram')
    axes[1, col].fill_between(times, np.maximum(0, centroid[0] - spec_bw[0]), np.minimum(centroid[0] + spec_bw[0], sr/2), alpha=0.5, label='Centroid +- bandwidth')
    axes[1, col].plot(times, centroid[0], label='Spectral centroid', color='w')


for i, file in enumerate(faulty_files + functioning_files):
    plot_file(file, i, axes=axes, fig=fig)


#y, sr = librosa.load(functioning_files[1])

#sd.play(y, sr)
#sd.wait()


plt.show()
