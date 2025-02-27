# ITAB COMPUTER SOFTWARE

## Objective
This Python program facilitates:

Communication with the ITAB gate via API calls.
Communication with the MCU to initiate the recording process over UART.
Simulation functions for both the API and UART communication, allowing you to test the MCU recording process without requiring access to an actual ITAB gate.
For details on how to use the simulation functions, refer to the main function in the source code.

## How to Run

1. Ensure you have followed the setup instructions in SETUP.md.
2. Activate the virtual environment:
```sh
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate   # On Windows
```

3. Run the program:
```sh
python RecordingController.py
```

4. The program will prompt you to:
* Select the gate type (1 or 0) where 1 is good-gate and 0 is faulty-gate.
* Enter the number of sequences to run (0 - 127).

5. Wait till program has completed.