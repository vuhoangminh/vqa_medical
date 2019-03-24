import sys
import os
import psutil
process = psutil.Process(os.getpid())


def print_section(print_string):
    print("\n"*2)
    print("="*80)
    print("working on:", print_string)
    print("="*80)


def print_processing(print_string):
    print(">> processing", print_string)


def print_separator():
    print("\n")
    print("-"*80)


def print_tqdm(index, total_len, cutoff=10, count=None):
    if index % cutoff == 0:
        if count is None:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %
                             (index, total_len, index*100.0/total_len))
            sys.stdout.flush()
        else:
            sys.stdout.write("processing %d/%d (%.2f%% done) -- added %d/%d  \r" %
                             (index, total_len, index*100.0/total_len, count, total_len))
            sys.stdout.flush()


def print_memory(print_string=''):
    print('Total memory in use ' + print_string + ': ', round(process.memory_info().rss/(2**30),2), ' GB')