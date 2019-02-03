import pickle



def save(obj, path):
    r"""Saves the object at the specified path."""
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)



def load(path):
    r"""Loads the object at the specified path."""
    with open(path, "rb") as fh:
        obj = pickle.load(fh)

    return obj
