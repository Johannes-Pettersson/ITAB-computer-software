import os


def get_files(num_of_files):
    """
    Set pick_randomly to False to get the same files every time

    Returns
        files: A list containing the good gate files
    """
    files = []

    training_data_dir = "Exjobb/Training_Files"

    for entry in os.scandir(training_data_dir):
        if entry.is_file():
            files.append(entry.path)
        if len(files) >= num_of_files:
            break

    if (len(files) != num_of_files):
        raise ValueError("Number of files in the directories is less than the number of files requested")

    return files