# ITAB COMPUTER SOFTWARE


## Project Overview


## Folder Structure

> [!WARNING]\
> This project is a `monorepo`, each folder is a different project.

```bash
.
├── README.md
├── Recording/
├── FeatureSelection/
└── System/
```

- **Recording**: Contains the python project to record Itab's Sigma Gate 2.
- **FeatureSelection**: Contains the python project to analyze different features from audio recorded from the gates.
- **System**: Contains the python project for the anomaly detection system, which is the main focus for this study. 

## Installation

### Clone the repository

Open a terminal and run the following commands:
```sh
git clone https://github.com/Johannes-Pettersson/ITAB-computer-software.git
cd ITAB-computer-software
git checkout master
git pull
```

### Create and use a Virtual Environment
To create a virtual environment, go to each folder(**FeatureSelection**, **Recording**, **System**) and run the following command. This will create a new virtual environment in a local folder named .venv:

> [!NOTE]\
> Project uses python version 3.12.

Create a virtual environment
```sh
python3.12 -m venv .venv # Unix/macOS
py -m venv .venv # Windows
```

Activate a Virtual Environment
```sh
source .venv/bin/activate # Unix/macOS
.venv\Scripts\activate # Windows
```

Install packages from requirements file
```sh
python3 -m pip install -r requirements.txt # Unix/macOS
py -m pip install -r requirements.txt # Windows
```

How to install a package inside .venv
```sh
python3 -m pip install example_package # Unix/macOS
py -m pip install example_package # Windows
```

Deactivate the environment when finished:
```sh
deactivate
```

### Additional Information

For further details on usage and additional configuration options, refer to the README.md file for each project.