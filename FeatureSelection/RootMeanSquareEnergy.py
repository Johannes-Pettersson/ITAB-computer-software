import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


print("Remote Mean Square Energy Begin")
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
# RMSE from scratch
def rms(signal, frame_length, hop_length):
    rmse = []
    for i in range(0, len(signal), hop_length):
        rms_current_frame = np.sqrt(np.sum(signal[i:i+frame_length]**2)/frame_length)
        rmse.append(rms_current_frame)
    return np.array(rmse)

# Function to load audio and compute envelope
def process_audio(file_path, frame_size, hop_size):
    y, sr = librosa.load(file_path)
    rmse = librosa.feature.rms(y=y, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)[0]
    frames = range(0, rmse.size)
    t = librosa.frames_to_time(frames, hop_length=HOP_SIZE)
    rmse_mean = np.mean(rmse)
    rmse_max = np.max(rmse)
    rmse_std = np.std(rmse)
    return y, sr, rmse, t, rmse_mean, rmse_max, rmse_std # Added mean, max, and std deviation

# Parameters
FRAME_SIZE = 512
HOP_SIZE = 512

# Process audio files
titles = [faulty_gates[1], faulty_gates[2], good_gates[1], good_gates[2]]
audio_data = [
    process_audio(titles[0], FRAME_SIZE, HOP_SIZE),
    process_audio(titles[1], FRAME_SIZE, HOP_SIZE),
    process_audio(titles[2], FRAME_SIZE, HOP_SIZE),
    process_audio(titles[3], FRAME_SIZE, HOP_SIZE)
]

# Set up the figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 7.25))

# Plot each waveform and amplitude envelope
for ax, (y, sr, rmse, t, rmse_mean, rmse_max, rmse_std), title in zip(axes.flatten(), audio_data, titles):
    ax.set_title(f"{title}\nMean: {rmse_mean:.4f}, Max: {rmse_max:.4f}, Std: {rmse_std:.4f}")
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.plot(t, rmse, color='r')

# Set consistent axis limits
max_time = max(t[-1] for _, _, _, t, _, _, _ in audio_data)
for ax in axes.flatten():
    ax.set_xlim([0, max_time])
    ax.set_ylim([-1, 1])  # Assuming normalized audio data
    
plt.tight_layout()
plt.show()

print("Remote Mean Square Energy End")