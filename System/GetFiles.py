import os
import random


def get_files(num_of_good_gate_files, num_of_faulty_gate_files, pick_randomly=True):
    """
    Set pick_randomly to False to get the same files every time

    Returns
    1. good_gate_files: A list containing the good gate files
    2. faulty_gate_files: A list containing the faulty gate files
    """
    good_gate_files = []
    faulty_gate_files = []

    functioning_directories = [
        "../Recording/Functioning_gate_recordings/Day 3/Session 1"
    ]
    faulty_directories = [

        "../Recording/Faulty_gate_recordings/Day 3/Session 1"
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
        good_gate_files.pop(random.randrange(len(good_gate_files))) if pick_randomly else good_gate_files.pop()

    while(len(faulty_gate_files) > num_of_faulty_gate_files):
        faulty_gate_files.pop(random.randrange(len(faulty_gate_files))) if pick_randomly else faulty_gate_files.pop()

    if ((len(good_gate_files) != num_of_good_gate_files) and (len(faulty_gate_files) != num_of_faulty_gate_files)):
        raise ValueError("Number of files in the directories is less than the number of files requested")

    return good_gate_files, faulty_gate_files

def get_files_from_single_dir(num_of_files, data_dir):
    """
    Set pick_randomly to False to get the same files every time

    Returns
        files: A list containing the good gate files
    """
    files = []

    for entry in os.scandir(data_dir):
        if entry.is_file():
            files.append(entry.path)
        if len(files) >= num_of_files:
            break

    if (len(files) != num_of_files):
        raise ValueError("Number of files in the directories is less than the number of files requested")

    return files

def get_files_from_directories(
    num_of_good_gate_files,
    num_of_faulty_gate_files,
    good_gate_dir,
    faulty_gate_dir,
    pick_randomly=True,
):
    """
    Set pick_randomly to False to get the same files every time

    Returns
    1. good_gate_files: A list containing the good gate files
    2. faulty_gate_files: A list containing the faulty gate files
    """
    good_gate_files = []
    faulty_gate_files = []

    if not os.path.isdir(good_gate_dir) or not os.path.isdir(faulty_gate_dir):
        raise ValueError("The directories do not exist")

    for entry in os.scandir(good_gate_dir):
        if entry.is_file():
            good_gate_files.append(entry.path)

    for entry in os.scandir(faulty_gate_dir):
        if entry.is_file():
            faulty_gate_files.append(entry.path)

    while len(good_gate_files) > num_of_good_gate_files:
        (
            good_gate_files.pop(random.randrange(len(good_gate_files)))
            if pick_randomly
            else good_gate_files.pop()
        )

    while len(faulty_gate_files) > num_of_faulty_gate_files:
        (
            faulty_gate_files.pop(random.randrange(len(faulty_gate_files)))
            if pick_randomly
            else faulty_gate_files.pop()
        )

    if (
        len(good_gate_files) != num_of_good_gate_files
        and len(faulty_gate_files) != num_of_faulty_gate_files
    ):
        raise ValueError(
            "Number of files in the directories is less than the number of files requested"
        )

    return good_gate_files, faulty_gate_files