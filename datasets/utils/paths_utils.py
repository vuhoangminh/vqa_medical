import os


def get_project_dir(path, project_name):
    paths = path.split(project_name)
    return paths[0] + project_name


def make_dir(dir):
    if not os.path.exists(dir):
        print("making dir", dir)
        os.makedirs(dir)
