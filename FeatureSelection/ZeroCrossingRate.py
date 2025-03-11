import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

print("Zero Crossing Rate Begin")
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

for i in range (len(faulty_gates)):
    # Load the faulty and good gate audio files
    y0, sr0 = librosa.load(faulty_gates[i])
    y1, sr1 = librosa.load(good_gates[i])

    # Convert 2 seconds to sample index (start counting from 2s onward)
    start_sample_0 = int(2 * sr0)  # For faulty gate
    start_sample_1 = int(2 * sr1)  # For good gate

    # Slice the audio starting from 2 seconds onward
    y0_trimmed = y0[:start_sample_0]
    y1_trimmed = y1[:start_sample_1]

    # Count zero crossings for the trimmed signal (from 2 seconds onward)
    zero_crossings_0 = librosa.zero_crossings(y0, pad=False)
    zero_crossings_1 = librosa.zero_crossings(y1, pad=False)

    # Print the results
    print(f"Zero Crossings {faulty_gates[i]}: {sum(zero_crossings_0)}")
    print(f"Zero Crossings {good_gates[i]}: {sum(zero_crossings_1)}")
    print("-------------------------------------------------------------------------------------")

title_gg = faulty_gates[8]
title_bg = faulty_gates[9]
y_good, sr_good = librosa.load(title_gg)
y_bad, sr_bad = librosa.load(title_bg)


# Compute Zero Crossing Rate (ZCR) for good gate
frame_length = 2048  # Frame size (larger = smoother)
hop_length = 512     # Step size between frames

zcr_good = librosa.feature.zero_crossing_rate(y_good, frame_length=frame_length, hop_length=hop_length)[0]
zcr_bad = librosa.feature.zero_crossing_rate(y_bad, frame_length=frame_length, hop_length=hop_length)[0]

# Convert frame indices to time for both signals
times_good = librosa.frames_to_time(np.arange(len(zcr_good)), sr=sr_good, hop_length=hop_length)
times_bad = librosa.frames_to_time(np.arange(len(zcr_bad)), sr=sr_bad, hop_length=hop_length)

zcr_limit = [0, max(np.max(zcr_good)*1.05, np.max(zcr_bad))*1.05]
waveform_limit = [min(y_good.min(), y_bad.min())*1.05, max(y_good.max(), y_bad.max())*1.05]
print("waveform_limit", waveform_limit)
print("zcr_limit", zcr_limit)

# Create a figure with 4 subplots (2 rows, 2 columns)
plt.figure(figsize=(15, 7.25))

# Subplot 1: Original waveform for good gate
plt.subplot(2, 2, 1)
librosa.display.waveshow(y_good, sr=sr_good, alpha=0.6, color='green')
plt.title(title_gg, fontsize=8)
plt.ylabel("Amplitude", fontsize=14)
plt.ylim(waveform_limit)
plt.grid(True)

# Subplot 2: Original waveform for bad gate
plt.subplot(2, 2, 2)
librosa.display.waveshow(y_bad, sr=sr_bad, alpha=0.6, color='red')
plt.title(title_bg, fontsize=8)
plt.ylim(waveform_limit)
plt.grid(True)

# Subplot 3: Combined Zero Crossing Rate (Good vs Bad)
plt.subplot(2, 1, 2)  # Merging 2 plots into 1 spanning full width
plt.plot(times_good, zcr_good, color='green', linewidth=2, label=title_gg)
plt.plot(times_bad, zcr_bad, color='red', linewidth=2, label=title_bg)
plt.title("Zero Crossing Rate Comparison", fontsize=16)
plt.xlabel("Time (seconds)", fontsize=14)
plt.ylabel("Zero Crossing Rate", fontsize=14)
plt.ylim(zcr_limit)
plt.grid(True)
plt.legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

print("Zero Crossing Rate End")