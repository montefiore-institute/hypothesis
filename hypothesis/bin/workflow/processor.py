import dill as pickle
import sys


def main():
    pickle.settings['recurse'] = True
    with open(sys.argv[1], "rb") as f:
        function = pickle.load(f)
    if len(sys.argv) > 2:
        task_index = int(sys.argv[2])
        function(task_index)
    else:
        function()


if __name__ == "__main__":
    main()
