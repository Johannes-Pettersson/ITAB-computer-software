

import os
from Demo import gate_sequence
import sys
import threading
import time

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)

from Recording.usb_mic import UsbRecorder

print("Retraining system")
num = 0
try:
    os.remove("training_data.pkl")
    print("Pickle file  deleted")
except FileNotFoundError:
    print("No pickle file found for deletion")

for _ in os.listdir("Training_Files"):
    num += 1

usb_recorder = UsbRecorder()
for i in range(num, 500):
    file_name = f"Training_Files/G_G_{i}.WAV"
    gate_sequence_th = threading.Thread(target=gate_sequence, daemon=True)
    gate_sequence_th.start()
    usb_recorder.record(file_name)

    gate_sequence_th.join()
    time.sleep(1)

print("Retrain finished")
