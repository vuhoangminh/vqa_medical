import os
import ntpath
import shutil


def get_project_dir(path, project_name):
    paths = path.split(project_name)
    return paths[0] + project_name


def make_dir(dir):
    if not os.path.exists(dir):
        print("making dir", dir)
        os.makedirs(dir)


def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_filename_without_extension(path):
    filename = get_filename(path)
    return os.path.splitext(filename)[0]


def get_parent_dir(path):
    return os.path.abspath(os.path.join(path, os.pardir))


def delete_dir(path):
    print(">> delete", path)
    shutil.rmtree(path)
