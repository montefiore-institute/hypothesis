import os


def exists(f):
    def wrapper():
        return os.path.exists(f)

    return wrapper
