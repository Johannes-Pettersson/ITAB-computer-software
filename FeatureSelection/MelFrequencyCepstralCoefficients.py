import librosa
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# def calculate_mfcc(file, dct_type=2, hop_length=512):
#     """Load audio and compute MFCC"""
#     y, sr = librosa.load(file, sr=None)  
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, dct_type=dct_type, hop_length=hop_length)
#     return mfcc  # Shape: (n_mfcc, time_steps)

# def compute_mfcc_statistics(mfcc):
#     """Compute statistical measures for each MFCC coefficient"""
#     return {
#         "mean": np.mean(mfcc, axis=1),
#         "std": np.std(mfcc, axis=1),
#         "skewness": scipy.stats.skew(mfcc, axis=1),
#         "kurtosis": scipy.stats.kurtosis(mfcc, axis=1)
#     }

# def plot_mfcc_comparison(statistic, stats_good, stats_faulty, dct_types, labels):
#     """Plot comparison of MFCC statistic for both good and faulty gates for all DCT types in a 2x2 grid"""
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows, 2 columns

#     for idx, dct_type in enumerate(dct_types):
#         ax = axes[idx // 2, idx % 2]  # Get correct subplot position

#         x = np.arange(len(labels))  # X-axis (MFCC coefficients)
#         width = 0.35  # Bar width for good vs faulty comparison

#         # Plot for good gates
#         ax.bar(x - width/2, stats_good[dct_type][statistic], width=width, label=f"Good Gates (DCT {dct_type})", color='blue')
#         mean_good = np.mean(stats_good[dct_type][statistic])
#         ptp_good = np.ptp(stats_good[dct_type][statistic])
#         max_good = np.max(stats_good[dct_type][statistic])
#         std_good = np.std(stats_good[dct_type][statistic])
#         ax.axhline(mean_good, color='blue', linestyle='--', label=f"Mean: {mean_good:.2f}")
#         ax.axhline(max_good, color='blue', linestyle='-.', label=f"Max: {max_good:.2f}")
#         # Plot for faulty gates
#         ax.bar(x + width/2, stats_faulty[dct_type][statistic], width=width, label=f"Faulty Gates (DCT {dct_type})", color='red', alpha=0.7)
#         mean_faulty = np.mean(stats_faulty[dct_type][statistic])
#         ptp_faulty = np.ptp(stats_faulty[dct_type][statistic])
#         max_faulty = np.max(stats_faulty[dct_type][statistic])
#         std_faulty = np.std(stats_faulty[dct_type][statistic])
#         ax.axhline(mean_faulty, color='red', linestyle='--', label=f"Mean: {mean_faulty:.2f}")
#         ax.axhline(max_faulty, color='red', linestyle='-.', label=f"Max: {max_faulty:.2f}")
#         print(f"Ptp Good: {ptp_good:.2f}, Ptp Faulty: {ptp_faulty:.2f}")

#         ax.set_xlabel("MFCC Coefficients")
#         ax.set_ylabel(statistic.capitalize())
#         ax.set_title(f"DCT Type {dct_type}: {statistic.capitalize()} for Good vs Faulty Gates")
#         ax.set_xticks(x)
#         # ax.set_xticklabels(labels)
#         ax.legend()

#     plt.tight_layout()  # Adjust layout to prevent overlap
#     plt.show()

# def calculate_values(file):
#     """
#     """
#     frame_size = 1024
#     hop_size = 512
#     dct_type = 2
#     y, sr = librosa.load(file)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, dct_type=dct_type, hop_length=hop_size)

# def main():
#     good_file = "../Recording/Functioning_gate_recordings/Day 2/Session 1/G_G_1.WAV"
#     faulty_file = "../Recording/Faulty_gate_recordings/Day 2/Session 1/B_G_1.WAV"

#     selected_statistic = "skewness"  # Choose from: "mean", "std", "skewness", "kurtosis"
#     dct_types = [1, 2, 3, 4]  # DCT types to compare

#     # Compute MFCC and statistics for good and faulty files
#     mfcc_good = {dct_type: calculate_mfcc(good_file, dct_type) for dct_type in dct_types}
#     mfcc_faulty = {dct_type: calculate_mfcc(faulty_file, dct_type) for dct_type in dct_types}

#     stats_good = {dct_type: compute_mfcc_statistics(mfcc) for dct_type, mfcc in mfcc_good.items()}
#     stats_faulty = {dct_type: compute_mfcc_statistics(mfcc) for dct_type, mfcc in mfcc_faulty.items()}

#     # Generate labels for MFCC coefficients
#     labels = [f"MFCC {i}" for i in range(1, len(stats_good[1][selected_statistic]) + 1)]

#     # Plot the selected statistic for good vs faulty gates for each DCT type in a 2x2 grid
#     plot_mfcc_comparison(selected_statistic, stats_good, stats_faulty, dct_types, labels)

# # Function to load and process audio
# def calculate_values(file, hop_length=512):
#     y, sr = librosa.load(file, sr=None)  # Load with original sample rate
    
#     # Compute MFCCs for all four DCT types
#     mfccs = {
#         1: librosa.feature.mfcc(y=y, sr=sr, dct_type=1, hop_length=hop_length),
#         2: librosa.feature.mfcc(y=y, sr=sr, dct_type=2, hop_length=hop_length),
#         3: librosa.feature.mfcc(y=y, sr=sr, dct_type=3, hop_length=hop_length),
#         4: librosa.feature.mfcc(y=y, sr=sr, dct_type=4, hop_length=hop_length),
#     }
#     return y, sr, mfccs, hop_length  # Return hop_length for correct timing

# def main():
#     titles = [
#         "../Recording/Functioning_gate_recordings/Day 2/Session 1/G_G_1.WAV",
#         "../Recording/Faulty_gate_recordings/Day 2/Session 1/B_G_1.WAV",
#     ]
#     audio_data = [calculate_values(file) for file in titles]

#     dct_types = [1, 2, 3, 4]  # DCT types to test

#     # Create subplots: 4 columns for each DCT type, one row per audio file
#     fig, axes = plt.subplots(len(titles), len(dct_types), figsize=(20, len(titles) * 4))
#     mng = plt.get_current_fig_manager()
#     mng.full_screen_toggle()

#     for ax_row, (y, sr, mfccs, hop_length), title in zip(axes, audio_data, titles):
#         for ax, dct_type in zip(ax_row, dct_types):
#             # Compute correct time axis
#             time_axis = librosa.frames_to_time(np.arange(mfccs[dct_type].shape[1]), sr=sr, hop_length=hop_length)

#             # Display MFCC spectrogram for each DCT type
#             img = librosa.display.specshow(mfccs[dct_type], x_axis="time", y_axis="1-20", sr=sr, hop_length=hop_length, ax=ax)
#             ax.set_title(f"DCT Type {dct_type}\n{title}", fontsize=8)
#             fig.colorbar(img, ax=ax)

#     # Adjust layout and show plots
#     plt.tight_layout()
#     plt.show()

def calculate_values(file, hop_length=512, coef=1, dct_type=4):
    """
    Calculate MFCC and statistical measures for a specific coefficient and DCT type.
    Using librosa and scipy.stats for MFCC and statistics computation.
    Parameters
    ----------
    file : str
        Path to the audio file.
    hop_length : int
        Hop length for computing MFCC.
    coef : int
        Coefficient to extract from the MFCC matrix.
    dct_type : int
        DCT type to use for computing MFCC.

    Returns
    -------
    y : np.ndarray
        Audio time series.
    sr : int
        Sampling rate of the audio.
    mfcc : np.ndarray
        MFCC matrix.
    mfcc_skewness : np.ndarray
        Skewness of the selected MFCC coefficient.
    mfcc_kurtosis : np.ndarray
        Kurtosis of the selected MFCC coefficient.  
    """
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, dct_type=dct_type, hop_length=hop_length)
    mfcc_subset = mfcc[coef-1:coef, :]
    mfcc_skewness = scipy.stats.skew(mfcc_subset, axis=1)
    mfcc_skewness = np.float32(mfcc_skewness.item())
    mfcc_kurtosis = scipy.stats.kurtosis(mfcc_subset, axis=1).astype(np.float32)
    mfcc_kurtosis = np.float32(mfcc_kurtosis.item())
    # print(f"coef: {coef}, dct_type: {dct_type}, kurtosis: {mfcc_kurtosis}")
    return y, sr, mfcc, mfcc_skewness, mfcc_kurtosis

# Function to load audio and compute MFCC
def calculate_mfcc(file, dct_type=2, hop_length=512):
    y, sr = librosa.load(file, sr=None)  
    mfcc = librosa.feature.mfcc(y=y, sr=sr, dct_type=dct_type, hop_length=hop_length)
    return mfcc  # Shape: (n_mfcc, time_steps)

# Compute statistical measures for only MFCC 1 & 2
def compute_mfcc_statistics(mfcc):
    """Compute statistics for only MFCC coefficient 1 and 2"""
    mfcc_subset = mfcc[:2, :]  # Extract MFCC 1 and 2 only
    return {
        "mean": np.mean(mfcc_subset, axis=1),
        "std": np.std(mfcc_subset, axis=1),
        "skewness": scipy.stats.skew(mfcc_subset, axis=1),
        "kurtosis": scipy.stats.kurtosis(mfcc_subset, axis=1)
    }

# Function to plot the statistical comparisons
def plot_mfcc_comparison(stats_good, stats_faulty, statistic, dct_types):
    fig, axes = plt.subplots(1, len(dct_types), figsize=(18, 5))  # One row, one subplot per DCT

    x_labels = ["MFCC 1", "MFCC 2"]  # Only comparing first two coefficients
    x = np.arange(len(x_labels))  # X-axis positions

    for idx, dct_type in enumerate(dct_types):
        ax = axes[idx]
        width = 0.35  # Bar width
        
        # Plot bars for good and faulty gates
        ax.bar(x - width/2, stats_good[dct_type][statistic], width=width, label="Good Gates", color='blue')
        ax.bar(x + width/2, stats_faulty[dct_type][statistic], width=width, label="Faulty Gates", color='red', alpha=0.7)

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(statistic.capitalize())
        ax.set_title(f"DCT {dct_type}: {statistic.capitalize()}")
        ax.legend()

    plt.tight_layout()
    plt.show()

# Main function
def main():
    good_file = "../Recording/Functioning_gate_recordings/Day 2/Session 1/G_G_1.WAV"
    faulty_file = "../Recording/Faulty_gate_recordings/Day 2/Session 1/B_G_1.WAV"

    selected_statistic = "kurtosis"  # Choose: "mean", "std", "skewness", "kurtosis"
    dct_types = [1, 2, 3, 4]  # DCT types to compare

    # Compute MFCC and statistics for good and faulty files
    mfcc_good = {dct_type: calculate_mfcc(good_file, dct_type) for dct_type in dct_types}
    mfcc_faulty = {dct_type: calculate_mfcc(faulty_file, dct_type) for dct_type in dct_types}

    stats_good = {dct_type: compute_mfcc_statistics(mfcc) for dct_type, mfcc in mfcc_good.items()}
    stats_faulty = {dct_type: compute_mfcc_statistics(mfcc) for dct_type, mfcc in mfcc_faulty.items()}
    calculate_values(faulty_file, dct_type=4, coef=1)
    # Print statistics for reference
    print("\n=== Statistical Comparison for MFCC 1 & 2 ===")
    for dct_type in dct_types:
        print(f"\nDCT Type {dct_type}:")
        for key in stats_good[dct_type].keys():
            print(f"{key.capitalize()} - Good: {stats_good[dct_type][key].round(3)}, Faulty: {stats_faulty[dct_type][key].round(3)}")

    # Plot the selected statistic
    plot_mfcc_comparison(stats_good, stats_faulty, selected_statistic, dct_types)

if __name__ == "__main__":
    main()

## Variables skewness, kurtosis, dct_types: 1, 2, 3, 4, mfcc-coefficients: 1, 2 are good