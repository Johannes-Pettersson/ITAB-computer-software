import os
import random
import shutil


def copy_and_rename_files(good_gate_files, faulty_gate_files):
    destination_dir = "goodGates"
    os.makedirs(destination_dir, exist_ok=True)

    for index, file in enumerate(good_gate_files):
        new_filename = f"G_G_{index}.WAV"
        new_filepath = os.path.join(destination_dir, new_filename)
        shutil.copy2(file, new_filepath)

    destination_dir = "faultyGates"
    os.makedirs(destination_dir, exist_ok=True)

    for index, file in enumerate(faulty_gate_files):
        new_filename = f"F_G_{index}.WAV"
        new_filepath = os.path.join(destination_dir, new_filename)
        shutil.copy2(file, new_filepath)


def copy_files_to_directories(category, data_type, file_list, iterations, num_of_files):
    if len(file_list) > num_of_files * iterations:
        for i in range(iterations):
            for j in range(num_of_files):
                file = random.choice(
                    file_list
                )  # Ensures that the files is randomly selected
                file_list.remove(file)
                destination_dir = f"../Dataset/{category}/{data_type}{i}/"
                os.makedirs(dir, exist_ok=True)
                shutil.copy(file, destination_dir)
    else:
        raise ValueError("Not enough files in the list")

    return file_list


def get_files(good_gate_dir, faulty_gate_dir):
    good_gate_files = []
    faulty_gate_files = []

    for entry in os.scandir(good_gate_dir):
        if entry.is_file():
            good_gate_files.append(entry.path)

    for entry in os.scandir(faulty_gate_dir):
        if entry.is_file():
            faulty_gate_files.append(entry.path)

    return good_gate_files, faulty_gate_files


def main():

    good_gate_files, faulty_gate_files = get_files("goodGates", "faultyGates")

    good_gate_files = copy_files_to_directories(
        "Training", "G_G_F_", good_gate_files, 5, 50
    )
    good_gate_files = copy_files_to_directories(
        "Training", "C_G_F_", good_gate_files, 5, 25
    )
    good_gate_files = copy_files_to_directories(
        "Evaluation", "G_G_F_", good_gate_files, 5, 50
    )
    good_gate_files = copy_files_to_directories(
        "Configuration", "G_G_F_", good_gate_files, 1, 50
    )

    faulty_gate_files = copy_files_to_directories(
        "Training", "F_G_F_", faulty_gate_files, 5, 50
    )
    faulty_gate_files = copy_files_to_directories(
        "Training", "C_G_F_", faulty_gate_files, 5, 25
    )
    faulty_gate_files = copy_files_to_directories(
        "Evaluation", "F_G_F_", faulty_gate_files, 5, 50
    )
    faulty_gate_files = copy_files_to_directories(
        "Configuration", "F_G_F_", faulty_gate_files, 1, 50
    )

    print(len(good_gate_files))
    print(len(faulty_gate_files))


if __name__ == "__main__":
    main()
