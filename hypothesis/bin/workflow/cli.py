import argparse
import hypothesis as h
import os
import sys


def main():
    initialize()


def initialize():
    os.makedirs(os.path.expanduser('~') + "/.hypothesis/config")


if __name__ == "__main__":
    main()
