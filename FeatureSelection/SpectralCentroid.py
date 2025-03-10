import librosa
import sounddevice
import matplotlib.pyplot as plt
import numpy as np


file1_path = "../Recording/Faulty_gate_recordings/Session 1/B_G_2.WAV"
file2_path = "../Recording/Functioning_gate_recordings/Session 2/G_G_1.WAV"


# Plot 1 calculations ---------------------------------------------------------------------

y1, sr1 = librosa.load(file1_path)

S1, phase1 = librosa.magphase(librosa.stft(y1))

cent1 = librosa.feature.spectral_centroid(S=S1)

times1 = librosa.times_like(cent1)

# Plot 2 calculations ----------------------------------------------------------------------

y2, sr2 = librosa.load(file2_path)

S2, phase2 = librosa.magphase(librosa.stft(y2))

cent2 = librosa.feature.spectral_centroid(S=S2)

times2 = librosa.times_like(cent2)

# ----------------------------------------------------------------------------------

# Create plot figure
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 8))

# Show Plot 1
librosa.display.specshow(librosa.amplitude_to_db(S1, ref=np.max), y_axis='log', x_axis='time', ax=ax1)
ax1.plot(times1, cent1.T, label="Spectral centroid", color='w')
ax1.legend(loc='upper right')
ax1.set(title='Log Power Spectrogram - File 1')

# Show Plot 2
librosa.display.specshow(librosa.amplitude_to_db(S2, ref=np.max), y_axis='log', x_axis='time', ax=ax2)
ax2.plot(times2, cent2.T, label="Spectral centroid", color='w')
ax2.legend(loc='upper right')
ax2.set(title='Log Power Spectrogram - File 2')


# Display plots
plt.tight_layout()
plt.show()

# sounddevice.play(y, sr)
# sounddevice.wait()