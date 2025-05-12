import os


def get_files(num_of_files, data_dir):
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