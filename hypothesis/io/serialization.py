"""
Serialization utilities for Hypothesis.
"""

import pickle



def save(object, path):
    with open(path, "wb") as fh:
        pickle.dump(object, fh)



def load(path):
    with open(path, "rb") as fh:
        object = pickle.load(path)

    return object
