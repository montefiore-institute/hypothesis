import argparse
import hypothesis as h
import os
import sys


def main():
    initialize()
    num_arguments = len(sys.argv)
    if "--list" in sys.argv:
        list_all_keys()
    elif num_arguments == 3:
        load_key(sys.argv[2])
    elif num_arguments == 4:
        store_key_value(sys.argv[2], sys.argv[3])
    else:
        show_help_and_exit()


def initialize():
    os.makedirs(os.path.expanduser('~') + "/.hypothesis/config")


def list_all_keys():
    raise Exception("Listing all keys not implemented")


def load_key(key):
    raise Exception("Loading key not implemented")


def store_key_value(key, value):
    raise Exception("Storing key value not implemented")


def show_help_and_exit():
    help = r"""hypothesis config [key] [value]

Usage:
  --list
    """
    print(help)
    sys.exit(0)


if __name__ == "__main__":
    main()
