# SYSTEM

## File structure

```bash
System/
├── README.md
├── requirements.txt
├── FeatureExtraction.py
├── LOF.py
├── ZScore.py
├── GetFiles.py
└── System.py
```

- **FeatureExtraction**: Contains the featureExtraction class, used to hold all feature values that is extracted from a audio recording.
- **GetFiles**: This could be used to retrieve all files from a selected folder.
- **System**: Contains functionality to do a single prediction, where you can chose what training data should be used for the system.
- **ZScore**: Contain calculation and plotting of the ZScore algorithm.
- **LOF**: Contains the calculation and plotting of the Local Outlier Factor algorithm.

## Installation

Follow the installation guidelines for python venv from root directory [README.md](../README.md).