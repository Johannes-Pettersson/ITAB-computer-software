

import os
from Demo import gate_sequence
import sys
import threading

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)

from Recording.usb_mic import UsbRecorder

print("Retraining system")
try:
    os.remove("training_data.pkl")
    print("Pickle file deleted")
except FileNotFoundError:
    print("No pickle file found for deletion")

usb_recorder = UsbRecorder()
for i in range(50):
    file_name = f"Training_Files/G_G_{i}.WAV"
    gate_sequence_th = threading.Thread(target=gate_sequence, daemon=True)
    gate_sequence_th.start()
    usb_recorder.record(file_name)

    gate_sequence_th.join()

print("Retrain finished")
