import librosa
import matplotlib.pyplot as plt
import numpy as np
# import sounddevice as sd

def calculate_values(file, roll_percent=.37):
    y, sr = librosa.load(file)

    S, phase = librosa.magphase(librosa.stft(y))

    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=roll_percent)

    ro_max = np.amax(rolloff)
    ro_min = np.amin(rolloff)
    ro_mean = np.mean(rolloff)
    ro_std = np.std(rolloff)
    
    return S, sr, rolloff, roll_percent, ro_max, ro_min, ro_mean, ro_std


def plot_file(file, row, col, axes):
    
    S, sr, rolloff, roll_percent, _, _, _, _ = calculate_values(file)

    times = librosa.times_like(rolloff)

    ax = axes[row, col] 
    
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
    ax.plot(times, rolloff[0], label=f"Roll-off frequency ({roll_percent})", color='w')
    ax.legend(loc='upper right')
    title = file.split("/")[-2] + "/" + file.split("/")[-1]
    ax.set_title(f"{title}")  

    if col != 0:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)


def main():

    faulty_gate_files = [
        "../Recording/Faulty_gate_recordings/Day 2/Session 1/B_G_3.WAV",
        "../Recording/Faulty_gate_recordings/Day 2/Session 1/B_G_56.WAV"
    ]

    good_gate_files = [
        "../Recording/Functioning_gate_recordings/Day 2/Session 2/G_G_15.WAV",
        "../Recording/Functioning_gate_recordings/Day 2/Session 2/G_G_124.WAV"
    ]


    fig, axes = plt.subplots(nrows=2, ncols=max(len(faulty_gate_files), len(good_gate_files)), figsize=(12, 8), sharex=True, sharey=True)


    for i, file in enumerate(faulty_gate_files):
        plot_file(file, 0, i, axes=axes)

    for i, file in enumerate(good_gate_files):
        plot_file(file, 1, i, axes=axes)


    #y, sr = librosa.load(good_gate_files[1])

    #sd.play(y, sr)
    #sd.wait()


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()