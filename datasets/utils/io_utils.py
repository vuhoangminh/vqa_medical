import pickle
import datetime


def get_string_datetime():
    now = datetime.datetime.now()
    if now.month < 10:
        month_string = '0'+str(now.month)
    else:
        month_string = str(now.month)
    if now.day < 10:
        day_string = '0'+str(now.day)
    else:
        day_string = str(now.day)
    yearmonthdate_string = str(now.year) + month_string + day_string
    return yearmonthdate_string


def write_list_to_file(my_list, path):
    with open(path, 'w+') as f:
        for item in my_list:
            f.write("%s\n" % item)


def read_file_to_list(path):
    with open(path, 'r') as f:
        x = f.readlines()
    return x


def write_pickle(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data
