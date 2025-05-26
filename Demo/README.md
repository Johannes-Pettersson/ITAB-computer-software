# DEMO

## File structure

```bash
System/
├── requirements.txt
├── README.md
├── Input_Files/
├── Training_Files/
├── Demo.py
├── Gate_Light_Controller.py
├── training_data.pkl
└── Retrain.py
```

- **Input_Files**: Here is where the input file should be located, to be classified.
- **Training_Files**: Here the training files should be stored, in order for the system to detect those.
- **Demo**: This is where the functionality to start the demo.
- **Gate_Light_Controller**: Contains the code responsible for the communication with the gate, gate sequence and the led controlling.
- **Training_data.pkl**: This is used to store the feature values for the training data, this way you dont have to calculate these for everytime you have a new input file, making the program run faster.
- **Retrain**: Run this file when you want to retrain your system with new recordings.

## Installation

Follow the installation guidelines for python venv from root directory [README.md](../README.md).

## How to run
Just do:
```bash
python Demo.py
```
to run the demo program, and if you want to retrain the system you will have to do:
```bash
python retrain.py
```