import librosa
import numpy as np
import scipy.stats

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
    return y, sr, mfcc, mfcc_skewness, mfcc_kurtosis

def plot_mfcc_comparison(ax, stats_gg, stats_fg, statistic, dct_type, coef):

    x_labels = [f"Coefficient: {coef}"]
    x = np.arange(len(x_labels))
    width = 0.35
        
    stats_gg = np.array(stats_gg).flatten()
    stats_fg = np.array(stats_fg).flatten()

    print(f"stats_gg: {stats_gg}")
    print(f"stats_fg: {stats_fg}")

    ax.bar(x - width/2, stats_gg, width=width, label="Good Gate", color='blue')
    ax.bar(x + width/2, stats_fg, width=width, label="Faulty Gate", color='red', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(statistic.capitalize())
    ax.set_title(f"DCT: {dct_type}")
    ax.legend()

def main():
    import matplotlib.pyplot as plt
    files = [
        "../Recording/Functioning_gate_recordings/Day 2/Session 1/G_G_1.WAV",
        "../Recording/Faulty_gate_recordings/Day 2/Session 1/B_G_1.WAV",
    ]

    # Parameters
    dct_type = 3
    coef = 2
    hop_length = 512
    statistics = ["skewness", "kurtosis"]

    audio_data = [calculate_values(file=file, hop_length=hop_length, coef=coef, dct_type=dct_type) for file in files]
    
    fig, axes = plt.subplots(2, 2)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    for ax, (y, sr, mfcc, mfcc_skewness, mfcc_kurtosis), file in zip(axes.flatten(), audio_data, files):
        ax.set_title(f"{file}\n")
        img = librosa.display.specshow(mfcc, x_axis="time", sr=sr, hop_length=hop_length, ax=ax)
        ax.set_title(f"{file}", fontsize=8)
        fig.colorbar(img, ax=ax)

    mfcc_gg_skewness = audio_data[0][3]
    mfcc_fg_skewness = audio_data[1][3]
    mfcc_gg_kurtosis = audio_data[0][4]
    mfcc_fg_kurtosis = audio_data[1][4]

    plot_mfcc_comparison(axes[1, 0], mfcc_gg_skewness, mfcc_fg_skewness, statistics[0], dct_type, coef)
    plot_mfcc_comparison(axes[1, 1], mfcc_gg_kurtosis, mfcc_fg_kurtosis, statistics[1], dct_type, coef)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()