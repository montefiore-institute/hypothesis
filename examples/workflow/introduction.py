import hypothesis as h
import hypothesis.workflow as w
import os


@w.root
def main():
    print("This is the root of the workflow!")
    print("This task will be executed first!")
    print("\n")


@w.dependency(main)
def after_main():
    print("Will only be executed after main has completed!")
    print("\n")


@w.dependency(after_main)
@w.dependency(main)
def after_after_main():
    print("You can add multiple dependencies like this.")
    print("You don't have to think about it.'")
    print("The computational graph will be verified and restructured!")
    print("\n")


@w.dependency(main)
@w.tasks(2)
def parallel(task_index):
    print("You can parallelize a task with a dependency!")
    print("Executing", task_index)
    if task_index == 1:
        print("\n")


@w.dependency(after_main)
@w.postcondition(w.exists("a_directory"))
def generate_a_dir():
    print("This task, and its parents, will only be executed whenever the postcondition has not been satisified.")
    os.makedirs("a_directory")


@w.dependency(generate_a_dir)
def done():
    print("All done!\n")


@w.dependency(after_after_main)
@w.postcondition(w.exists("a_second_directory"))
def generate_a_dir_2():
    print("This task, and its parents, will only be executed whenever the postcondition has not been satisified.")
    os.makedirs("a_second_directory")


from hypothesis.workflow.local import execute
execute()
