Module hypothesis.util.base
===========================
General utilities for :mod:`hypothesis`.

Functions
---------

    
`is_integer(item)`
:   Checks whether the specified item is an integer.
    
    :param item: Any possible Python instance.
    :rtype: bool

    
`is_iterable(item)`
:   Checks whether the specified item is iterable.
    
    :param item: Any possible Python instance.
    :rtype: bool

    
`load_module(full_modulename)`
:   Loads the specified module (or class).
    
    :param full_modulename: The full module name of the method, class
                            or variable to load.