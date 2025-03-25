import numpy as np
from scipy.stats import zscore
from  Features.RootMeanSquareEnergy import calculate_values as rmse_calculate_values
import os

def get_files(num_of_good_gate_files, num_of_faulty_gate_files):
    good_gate_files = []
    faulty_gate_files = []

    functioning_directories = [
        "../Recording/Functioning_gate_recordings/Day 2/Session 1",
        "../Recording/Functioning_gate_recordings/Day 2/Session 2"
    ]
    faulty_directories = [

        "../Recording/Faulty_gate_recordings/Day 2/Session 1",
        "../Recording/Faulty_gate_recordings/Day 2/Session 2"

    ]

    for dir in functioning_directories:
        for entry in os.scandir(dir):  
            if entry.is_file():
                good_gate_files.append(entry.path)

    for dir in faulty_directories:
        for entry in os.scandir(dir):  
            if entry.is_file():
                faulty_gate_files.append(entry.path)    

    while(len(good_gate_files) >  num_of_good_gate_files):
        good_gate_files.pop()

    while(len(faulty_gate_files) > num_of_faulty_gate_files):
        faulty_gate_files.pop()

    return good_gate_files, faulty_gate_files

def main():
    num_of_good_gate_files = 550
    num_of_faulty_gate_files = 500
    num_of_training_files = 50
    good_gate_files, faulty_gate_files = get_files(num_of_good_gate_files, num_of_faulty_gate_files)
    
    train_data = []
    input_data = []
    another_input_data = []
    for i in range(num_of_training_files):
        _, _, _, _, mean_val, max_val, std_val = rmse_calculate_values(good_gate_files[i])
        train_data.append(mean_val)
    
    for i in range(num_of_faulty_gate_files):
        _, _, _, _, mean_val, max_val, std_val = rmse_calculate_values(faulty_gate_files[i])
        input_data.append(mean_val)

    for i in range(num_of_training_files, num_of_good_gate_files):
        _, _, _, _, mean_val, max_val, std_val = rmse_calculate_values(good_gate_files[i])
        another_input_data.append(mean_val)

    # print(data)
    train_mean = np.mean(train_data)
    train_std_dev = np.std(train_data)
    train_z_score = (train_data - train_mean) / train_std_dev

    input_z_score = (input_data - train_mean) / train_std_dev
    another_input_zscore = (another_input_data - train_mean) / train_std_dev

    delta = 0.5
    t = 3.0
    threshold = max(abs(train_z_score)) + delta
    threshold = max(threshold, t)
    accuracy = 0

    for val in input_z_score:
        val = abs(val)
        if val > threshold:
            accuracy += 1
        else: 
            print("z_score faulty gates")
            print(val)
            print("^^^False Positive^^^")

    for val in another_input_zscore:
        val = abs(val)
        if val < threshold:
            accuracy += 1
        else:
            print("z_score good gates")
            print(val)
            print("^^^False Negative^^^")

    accuracy = (accuracy / (len(input_z_score) + len(another_input_zscore))) * 100
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy}%, for {num_of_good_gate_files-num_of_training_files} good gates and {num_of_faulty_gate_files} faulty gates")

if __name__ == "__main__":
    main()