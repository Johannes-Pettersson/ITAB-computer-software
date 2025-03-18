import librosa
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def compute_mfcc_statistics(mfcc):
    """Compute statistical measures for each MFCC coefficient"""
    return {
        "mean": np.mean(mfcc, axis=1),
        "std": np.std(mfcc, axis=1),
        "skewness": scipy.stats.skew(mfcc, axis=1),
        "kurtosis": scipy.stats.kurtosis(mfcc, axis=1)
    }

def plot_mfcc_comparison(statistic, stats_good, stats_faulty, dct_types, labels):
    """Plot comparison of MFCC statistic for both good and faulty gates for all DCT types in a 2x2 grid"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows, 2 columns

    for idx, dct_type in enumerate(dct_types):
        ax = axes[idx // 2, idx % 2]  # Get correct subplot position

        x = np.arange(len(labels))  # X-axis (MFCC coefficients)
        width = 0.35  # Bar width for good vs faulty comparison

        # Plot for good gates
        ax.bar(x - width/2, stats_good[dct_type][statistic], width=width, label=f"Good Gates (DCT {dct_type})", color='blue')
        mean_good = np.mean(stats_good[dct_type][statistic])
        ptp_good = np.ptp(stats_good[dct_type][statistic])
        max_good = np.max(stats_good[dct_type][statistic])
        std_good = np.std(stats_good[dct_type][statistic])
        ax.axhline(mean_good, color='blue', linestyle='--', label=f"Mean: {mean_good:.2f}")
        ax.axhline(max_good, color='blue', linestyle='-.', label=f"Max: {max_good:.2f}")
        # Plot for faulty gates
        ax.bar(x + width/2, stats_faulty[dct_type][statistic], width=width, label=f"Faulty Gates (DCT {dct_type})", color='red', alpha=0.7)
        mean_faulty = np.mean(stats_faulty[dct_type][statistic])
        ptp_faulty = np.ptp(stats_faulty[dct_type][statistic])
        max_faulty = np.max(stats_faulty[dct_type][statistic])
        std_faulty = np.std(stats_faulty[dct_type][statistic])
        ax.axhline(mean_faulty, color='red', linestyle='--', label=f"Mean: {mean_faulty:.2f}")
        ax.axhline(max_faulty, color='red', linestyle='-.', label=f"Max: {max_faulty:.2f}")
        print(f"Ptp Good: {ptp_good:.2f}, Ptp Faulty: {ptp_faulty:.2f}")

        ax.set_xlabel("MFCC Coefficients")
        ax.set_ylabel(statistic.capitalize())
        ax.set_title(f"DCT Type {dct_type}: {statistic.capitalize()} for Good vs Faulty Gates")
        ax.set_xticks(x)
        # ax.set_xticklabels(labels)
        ax.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def calculate_mfcc(file, dct_type=2, hop_length=512):
    """Load audio and compute MFCC"""
    y, sr = librosa.load(file, sr=None)  
    mfcc = librosa.feature.mfcc(y=y, sr=sr, dct_type=dct_type, hop_length=hop_length)
    return mfcc  # Shape: (n_mfcc, time_steps)

def calculate_values(file_path):
    """

    """
    frame_size = 1024
    hop_size = 512
    dct_type = 2
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, dct_type=dct_type, hop_length=hop_size)
    ae = amplitude_envelope(y, frame_size, hop_size)
    t = librosa.frames_to_time(range(len(ae)), hop_length=hop_size)
    mean_val = np.mean(ae)
    max_val = np.max(ae)
    std_val = np.std(ae)
    return y, sr, ae, t, mean_val, max_val, std_val

def main():
    good_file = "../Recording/Functioning_gate_recordings/Day 2/Session 1/G_G_1.WAV"
    faulty_file = "../Recording/Faulty_gate_recordings/Day 2/Session 1/B_G_1.WAV"

    selected_statistic = "skewness"  # Choose from: "mean", "std", "skewness", "kurtosis"
    dct_types = [1, 2, 3, 4]  # DCT types to compare

    # Compute MFCC and statistics for good and faulty files
    mfcc_good = {dct_type: calculate_mfcc(good_file, dct_type) for dct_type in dct_types}
    mfcc_faulty = {dct_type: calculate_mfcc(faulty_file, dct_type) for dct_type in dct_types}

    stats_good = {dct_type: compute_mfcc_statistics(mfcc) for dct_type, mfcc in mfcc_good.items()}
    stats_faulty = {dct_type: compute_mfcc_statistics(mfcc) for dct_type, mfcc in mfcc_faulty.items()}

    # Generate labels for MFCC coefficients
    labels = [f"MFCC {i}" for i in range(1, len(stats_good[1][selected_statistic]) + 1)]

    # Plot the selected statistic for good vs faulty gates for each DCT type in a 2x2 grid
    plot_mfcc_comparison(selected_statistic, stats_good, stats_faulty, dct_types, labels)

if __name__ == "__main__":
    main()
