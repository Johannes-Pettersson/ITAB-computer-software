import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


print("Amplitude Envelope Begin")
faulty_gates = [
    "../Recording/Faulty_gate_recordings/Session 1/B_G_1.WAV",
    "../Recording/Faulty_gate_recordings/Session 1/B_G_25.WAV",
    "../Recording/Faulty_gate_recordings/Session 1/B_G_50.WAV",
    "../Recording/Faulty_gate_recordings/Session 1/B_G_75.WAV",
    "../Recording/Faulty_gate_recordings/Session 1/B_G_99.WAV",
    "../Recording/Faulty_gate_recordings/Session 3/B_G_1.WAV",
    "../Recording/Faulty_gate_recordings/Session 3/B_G_25.WAV",
    "../Recording/Faulty_gate_recordings/Session 3/B_G_50.WAV",
    "../Recording/Faulty_gate_recordings/Session 3/B_G_75.WAV",
    "../Recording/Faulty_gate_recordings/Session 3/B_G_99.WAV",
]
good_gates = [
    "../Recording/Functioning_gate_recordings/Session 2/G_G_1.WAV",
    "../Recording/Functioning_gate_recordings/Session 2/G_G_25.WAV",
    "../Recording/Functioning_gate_recordings/Session 2/G_G_50.WAV",
    "../Recording/Functioning_gate_recordings/Session 2/G_G_75.WAV",
    "../Recording/Functioning_gate_recordings/Session 2/G_G_99.WAV",
    "../Recording/Functioning_gate_recordings/Session 2/G_G_101.WAV",
    "../Recording/Functioning_gate_recordings/Session 2/G_G_125.WAV",
    "../Recording/Functioning_gate_recordings/Session 2/G_G_150.WAV",
    "../Recording/Functioning_gate_recordings/Session 2/G_G_175.WAV",
    "../Recording/Functioning_gate_recordings/Session 2/G_G_199.WAV",
]
# Function to compute amplitude envelope
def amplitude_envelope(signal, frame_size, hop_size):
    amplitude_envelope = []
    for i in range(0, len(signal), hop_size):
        current_frame_amplitude_envelope = max(signal[i:i+frame_size])
        amplitude_envelope.append(current_frame_amplitude_envelope)
    return np.array(amplitude_envelope)

# Function to load audio and compute envelope
def process_audio(file_path, frame_size, hop_size):
    y, sr = librosa.load(file_path)
    ae = amplitude_envelope(y, frame_size, hop_size)
    t = librosa.frames_to_time(range(len(ae)), hop_length=hop_size)
    return y, sr, ae, t, np.mean(ae), np.max(ae), np.std(ae)  # Added mean, max, and std deviation

# Parameters
FRAME_SIZE = 512
HOP_SIZE = 512

# Process audio files
titles = [faulty_gates[4], faulty_gates[6], good_gates[4], good_gates[6]]
audio_data = [
    process_audio(titles[0], FRAME_SIZE, HOP_SIZE),
    process_audio(titles[1], FRAME_SIZE, HOP_SIZE),
    process_audio(titles[2], FRAME_SIZE, HOP_SIZE),
    process_audio(titles[3], FRAME_SIZE, HOP_SIZE)
]

# Set up the figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 7.25))

# Plot each waveform and amplitude envelope
for ax, (y, sr, ae, t, mean_ae, max_ae, std_ae), title in zip(axes.flatten(), audio_data, titles):
    ax.set_title(f"{title}\nMean: {mean_ae:.4f}, Max: {max_ae:.4f}, Std: {std_ae:.4f}")
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.plot(t, ae, color='r')

# Set consistent axis limits
max_time = max(t[-1] for _, _, _, t, _, _, _ in audio_data)
for ax in axes.flatten():
    ax.set_xlim([0, max_time])
    ax.set_ylim([-1, 1])  # Assuming normalized audio data
    
plt.tight_layout()
plt.show()

print("Amplitude Envelope End")