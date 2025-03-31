import os
import random


def get_files(
    num_of_good_gate_files,
    num_of_faulty_gate_files,
    good_gate_dir=None,
    faulty_gate_dir=None,
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

    functioning_directories = [
        (
            (
                "../Recording/Functioning_gate_recordings/Day 2/Session 1",
                "../Recording/Functioning_gate_recordings/Day 2/Session 2",
            )
            if good_gate_dir is None
            else good_gate_dir
        )
    ]
    faulty_directories = [
        (
            (
                "../Recording/Faulty_gate_recordings/Day 2/Session 1",
                "../Recording/Faulty_gate_recordings/Day 2/Session 2",
            )
            if faulty_gate_dir is None
            else faulty_gate_dir
        )
    ]
    print("-------------------")
    print(functioning_directories)
    print(faulty_directories)
    print("-------------------")
    if not os.path.isdir(functioning_directories[0]) or not os.path.isdir(
        faulty_directories[0]
    ):
        raise ValueError("The directories do not exist")
    else:
        print("Directories exist")

    for dir in functioning_directories:
        for entry in os.scandir(dir):
            if entry.is_file():
                good_gate_files.append(entry.path)

    for dir in faulty_directories:
        for entry in os.scandir(dir):
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

    print("OK!!")

    return good_gate_files, faulty_gate_files
