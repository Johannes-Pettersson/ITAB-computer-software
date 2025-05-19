import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

class UsbRecorder():
    def __init__(self):
        self.sample_rate = 48000
        self.channels = 1
        self.device_index = 0 # Check with sd.query_devices()

    def record(self, file_name, duration=6):
        print("Recording...")
        recording = sd.rec(int(self.sample_rate * duration),
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        dtype='int16',
                        device=self.device_index)
        sd.wait()
        wav.write(file_name, self.sample_rate, recording)
        print("Recording finished")

def main():
    recorder = UsbRecorder()

    recorder.record("tesagta.wav")

if __name__ == "__main__":
    main()