import argparse
import hypothesis as h
import os
import pickle
import sys


root = os.path.expanduser('~') + "/.hypothesis/config"
storage_path = root + "/storage.pickle"


def main():
    initialize()
    num_arguments = len(sys.argv)
    if "--list" in sys.argv:
        list_all_keys()
    if "--clear" in sys.argv:
        clear_storage()
    elif num_arguments == 3:
        load_key(sys.argv[2])
    elif num_arguments == 4:
        store_key_value(sys.argv[2], sys.argv[3])
    else:
        show_help_and_exit()


def initialize():
    global root
    global storage_path
    if not os.path.exists(root):
        os.makedirs(root)
    # Check if the data storage exists.
    if not os.path.exists(storage_path):
        with open(storage_path, "wb") as handle:
            pickle.dump({}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def list_all_keys():
    with open(storage_path, "rb") as handle:
        storage = pickle.load(handle)
    keys = list(storage.keys())
    keys.sort()
    for k in keys:
        print(k)


def clear_storage():
    global storage_path
    try:
        os.remove(storage_path)
    except:
        pass


def load_key(key):
    with open(storage_path, "rb") as handle:
        storage = pickle.load(handle)
    if key in storage.keys():
        print(storage[key])


def store_key_value(key, value):
    with open(storage_path, "rb") as handle:
        storage = pickle.load(handle)
    storage[key] = value
    with open(storage_path, "wb") as handle:
        pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)


def show_help_and_exit():
    help = r"""hypothesis config [key] [value]

Usage:
  --clear    Clears all stored key-value pairs.
  --list     Lists all stored keys.
    """
    print(help)
    sys.exit(0)


if __name__ == "__main__":
    main()
