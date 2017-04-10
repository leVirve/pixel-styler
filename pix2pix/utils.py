import os


def mkdir(folder, parent):
    full_path = os.path.join(parent, folder)
    os.makedirs(full_path, exist_ok=True)
    return full_path
