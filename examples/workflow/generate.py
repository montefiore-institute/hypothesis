import hypothesis as h
import hypothesis.workflow as w
import numpy as np
import os


@w.root
def main():
    print("This is the root of the workflow!")
    print("This task will be executed first!")


# It's better to do array tasks like this.

n = 100
parameters = np.arange(n)

@w.dependency(main)
@w.tasks(n)
def task(x):
    print("Executing task with the default way of doing things.", str(x))


# But you can also generate them like this

I = 10
J = 10

previous = task
for i in range(I):
    for j in range(J):
        @w.dependency(previous)  # Trick to generate the matrix in order :D
        def test(i=i, j=j):  # This how to force the software not to use the global reference.
            print("Computing matrix ", i, ",", j, "...")
        previous = test
