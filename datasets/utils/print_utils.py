def print_section(print_string):
    print("\n"*2)
    print("="*80)
    print("working on:", print_string)
    print("="*80)

def print_processing(print_string):
    print (">> processing", print_string)

def print_separator():
    print("\n")
    print("-"*80)    

def print_list(mylist):
    i=0
    while i < len(mylist):
        print(mylist[i])
        i += 1 