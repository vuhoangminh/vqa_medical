def write_list_to_file(my_list, path):
    with open(path, 'w+') as f:
        for item in my_list:
            f.write("%s\n" % item)


def read_file_to_list(path):
    with open(path, 'r') as f:
        x = f.readlines()
    return x