# SYSTEM

## Folder structure

```bash
System/
├── README.md
├── requirements.txt
├── FeatureExtraction.py
├── GetFiles.py
├── LOF.py
├── ZScore.py
└── System.py
```

- **LOF / ZScore**: Contains the calculations for each feature, one file for each feature.
- **System**: Contains the anomaly detection systems logic.
- **FeatureExtraction.py**: Handles the extraction of values from files provided and also the feature list provided.

## Installation

Follow the installation guidelines for python venv from root directory [README.md](../README.md).


## How to run the final test (Based on the research questions, link below)
[Link to research in diva-portal.org](https://hj.diva-portal.org/smash/record.jsf?pid=diva2%3A1967634&dswid=-2394) In order to fully understand details of this program, you must read the method part.

The system functionality can be found in the System.py file. I will briefly describe the main functionality. 
1. First we declare what features we are going to use for the test. (feature_list[])
2. We will go into a section that will loop 5 times, the 5 times are the number of replications for the final test.
3. Inside of this loop, we will declare the different folders where the audio files is stored. 
* G_G_F{num} Good Gate Files 0: for the training data
* F_G_F{num} Good Gate Files 0: -||-
* C_G_G_F{num} Good Gate Files 0 -||-
* C_F_G_F{num} Good Gate Files 0 -||-
* G_G_F: for the evaluation data
* F_G_F: for the evaluation data
4. Comments in the code describe the next few lines properly.