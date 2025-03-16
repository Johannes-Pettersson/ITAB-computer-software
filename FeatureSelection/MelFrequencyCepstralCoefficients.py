import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Example audio file paths (replace with your actual paths)
good_gates = [
    "../Recording/Functioning_gate_recordings/Day 2/Session 1/G_G_1.WAV",
    "../Recording/Functioning_gate_recordings/Day 2/Session 1/G_G_25.WAV",
    "../Recording/Functioning_gate_recordings/Day 2/Session 1/G_G_50.WAV",
    "../Recording/Functioning_gate_recordings/Day 2/Session 1/G_G_75.WAV",
    "../Recording/Functioning_gate_recordings/Day 2/Session 1/G_G_100.WAV",
]

faulty_gates = [
    "../Recording/Faulty_gate_recordings/Day 2/Session 1/B_G_1.WAV",
    "../Recording/Faulty_gate_recordings/Day 2/Session 1/B_G_25.WAV",
    "../Recording/Faulty_gate_recordings/Day 2/Session 1/B_G_50.WAV",
    "../Recording/Faulty_gate_recordings/Day 2/Session 1/B_G_75.WAV",
    "../Recording/Faulty_gate_recordings/Day 2/Session 1/B_G_100.WAV",
]


# Define the time windows for anomaly detection (in seconds)
windows = [(1.5, 2.0), (3.0, 3.5)]

# Function to process the audio files and extract Mel Spectrogram and MFCC
def process_audio(file_path):
    y, sr = librosa.load(file_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return y, sr, S_db, mfcc

# Process good and bad gates
audio_data_good = [process_audio(file) for file in good_gates]
audio_data_faulty = [process_audio(file) for file in faulty_gates]

# Create the figure for plotting
fig, axes = plt.subplots(len(windows) * 2, 2, figsize=(15, len(windows) * 8))

# Loop through each time window and each audio data (good and faulty)
for idx, (start_time, end_time) in enumerate(windows):
    # Extract the segment for each audio file and plot
    for i, (audio_data, title) in enumerate(zip([audio_data_good, audio_data_faulty], 
                                               ['Good Gates', 'Faulty Gates'])):
        for j, (y, sr, S_db, mfcc) in enumerate(audio_data):
            # Extract segment for specific time window
            start_sample = librosa.time_to_samples(start_time, sr=sr)
            end_sample = librosa.time_to_samples(end_time, sr=sr)

            # Ensure the indices are within bounds of the audio
            start_sample = max(start_sample, 0)
            end_sample = min(end_sample, len(y))

            print(f"Start sample: {start_sample}, End sample: {end_sample}, Audio length: {len(y)}")

            # Check if the segment is non-empty
            if end_sample > start_sample:
                # Extract the segment for Mel Spectrogram
                S_db_segment = S_db[:, start_sample:end_sample]
                mfcc_segment = mfcc[:, start_sample:end_sample]
            else:
                # If no valid segment, create an empty array
                S_db_segment = np.zeros((S_db.shape[0], 1))
                mfcc_segment = np.zeros((mfcc.shape[0], 1))

            # Check if the extracted segment is empty
            if S_db_segment.size == 0:
                print(f"Warning: Empty segment for {title} in time window ({start_time}s to {end_time}s)")

            # # Plot Mel Spectrogram
            # ax1 = axes[idx + i * len(windows), 0]  # Good gates on top, faulty gates on bottom
            # img1 = librosa.display.specshow(S_db_segment, x_axis='time', y_axis='mel', ax=ax1)
            # ax1.set_title(f'{title} Mel Spectrogram ({start_time}-{end_time}s) - File {j+1}')
            # fig.colorbar(img1, ax=ax1)

            # Plot MFCC
            ax2 = axes[idx + i * len(windows), 1]
            img2 = librosa.display.specshow(mfcc_segment, x_axis='time', ax=ax2)
            ax2.set_title(f'{title} MFCC ({start_time}-{end_time}s) - File {j+1}')
            fig.colorbar(img2, ax=ax2)

# Adjust layout
plt.tight_layout()
plt.show()
