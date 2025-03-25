# ITAB COMPUTER SOFTWARE - Recording

## Objective
This Python program facilitates:

Plottings for different feature extractions used throughout this study.
A `FeaturePlotting.py` program that lets you plott specified amount of files (both good and bad gate files) and lets you choose between one and two features to plot in a graph.

### Installation specifics for this project
1. Install packages from `requirements.txt` file.

### How to run

1. Ensure you have followed the installation guidelines for venv from [README.md](../README.md).
2. Activate the virtual environment:
```sh
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate   # On Windows
```

3. Run the program:
```sh
python FeaturePlotting.py --good_gate_files <number> --faulty_gate_files <number> --features <feature> <feature>
```

4. Plotting should now be seen based of the arguments provided when running the program.

5. For additional information you are able to run:
```sh
python FeaturePlotting.py -h 
```