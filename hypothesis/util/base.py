r"""General ``hypothesis`` utilities.

"""



def is_iterable(instance):
    try:
        iter(instance)
        iterable = True
    except TypeError:
        iterable = False

    return iterable
