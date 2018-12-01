"""
Serialization utilities for Hypothesis.
"""

import pickle



def save(obj, path):
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)



def load(path):
    with open(path, "rb") as fh:
        obj = pickle.load(fh)

    return obj
