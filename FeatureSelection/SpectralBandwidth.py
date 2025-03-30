import librosa
import matplotlib.pyplot as plt
import numpy as np
# import sounddevice as sd

def calculate_values(file_path):
    """
    Calculate the max, min and ptp-value of the spectral bandwidth of an audio file.
    Uses librosa to load the audio file and compute the Spectral bandwidth
    
    Parameters
    ----------
        file_path : str Path to the audio file.
        
    Returns
    -------        
        sr : sampling rate of y (librosa.load).
        
        S : Spectogram magnitude
        
        spec_bw : Spectral bandwidth
        
        bw_min : Minimum value of spectral bandwidth
        
        bw_ptp : Peak-to-Peak value of spectral bandwidth
        
        bw_max : maximum value of spectral bandwidth

        bw_mean : mean value of spectral bandwidth

        bw_std : standard deviation of spectral bandwidth
    """

    y, sr = librosa.load(file_path)
    S, phase = librosa.magphase(librosa.stft(y=y))
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, p=1)

    bw_min = np.min(spec_bw)
    bw_max = np.max(spec_bw)
    bw_ptp = np.ptp(spec_bw)
    bw_mean = np.mean(spec_bw)
    bw_std = np.std(spec_bw)

    return sr, S, spec_bw, bw_min, bw_ptp, bw_max, bw_mean, bw_std


def plot_file(file, col, axes, fig):
    sr, S, spec_bw, _, _, _, _, _ = calculate_values(file)

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


def main():

    faulty_gate_files = [
        "../Recording/Faulty_gate_recordings/Day 2/Session 1/B_G_100.WAV"
    ]

    good_gate_files = [
        "../Recording/Functioning_gate_recordings/Day 2/Session 1/G_G_419.WAV"
    ]

    fig, axes = plt.subplots(nrows=2, ncols=len(faulty_gate_files) + len(good_gate_files), figsize=(12, 8), sharex=True)


    for i, file in enumerate(faulty_gate_files + good_gate_files):
        plot_file(file, i, axes=axes, fig=fig)


    #y, sr = librosa.load(good_gate_files[1])

    #sd.play(y, sr)
    #sd.wait()

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    plt.show()

if __name__ == "__main__":
    main()
