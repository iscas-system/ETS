import os


def find_file_with_suffix(dir_path: str, suffix: str):
    """
    find file with suffix in dir_path
    :param dir_path:
    :param suffix:
    :return:
    """
    files = os.listdir(dir_path)
    for _, file in enumerate(files):
        if file.endswith(suffix):
            return file
    return Non


e