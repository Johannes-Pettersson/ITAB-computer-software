# ITAB COMPUTER SOFTWARE - Recording

## Objective
This Python program facilitates:

Communication with the ITAB gate via API calls.
Communication with the MCU to initiate the recording process over UART.
Simulation functions for both the API and UART communication, allowing you to test the MCU recording process without requiring access to an actual ITAB gate.
For details on how to use the simulation functions, refer to the main function in the source code.

### Installation specifics for this project
1. Install packages from `requirements.txt` file.
2. Create your own `.env`file.
```sh
cp .env.example .env
```
3. Make sure to correctly set the variables in the `.env`file.
- **UART_PORT**: Your usb port connected to the STM32 MCU.
- **GATE_URL**: IP Adress for the gate
- **GATE_ID**: Id for the gate


### How to run

1. Ensure you have followed the installation guidelines for venv from [README.md](../README.md).
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
* Enter the number of sequences to run (0 - 1000).
* The program will loop through all the sequences entered.

5. Wait till program has completed.