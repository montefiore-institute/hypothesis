# Workflows

## Usage

### Using the `hypothesis` CLI tool
> Relevant to the workflow in `simulate.py`

```console
you@local:~ $ hypothesis workflow execute simulate.py
Using the local workflow backend.
Executing root node.
Simulating training block 0
Simulating training block 1
Simulating training block 2
Simulating training block 3
Simulating training block 4
Simulating training block 5
Simulating training block 6
Simulating training block 7
...
```

Executing this a 2nd time only executes the main subroutine, as the software detects
which conditions in the computational graph have been satisfied already.

> **Recommendation** Put an `alias` in your `.bashrc` file to ease the usage.

```console
alias h='hypothesis'
alias w='hypothesis workflow'
```

> **Note**: The following assumes a Slurm enabled HPC cluster

```console
you@local:~ $ hypothesis workflow execute simulate.py
```

Custom options are available to set the directory to generate the scripts `--directory`,
the Slurm partition `--partition`, to cleanup the generated files `--cleanup`, or
a custom Anaconda environment `--environment`.

For instance, executing
```console
you@local:~ $ hypothesis workflow execute simulate.py --partition debug --directory job --environment hypothesis
Using Slurm backend.
Executing workflow tmpoyfosiru
```
will generate all files in the `job` directory, and allocate a workflow with the name `tmpoyfosiru`.
Custom names can be provided by specifying `--name`. Afterwards, jobs can be canceled or deleted (their respective Slurm jobs as well)
by executing `hypothesis workflow delete tmpoy`. It should be noted that the workflow name can be a substring.
Adding the `--parsable` option to the CLI arguments, ensures that the output is easily processable for
the usage in shell scripts.

## Example: simulation pipeline
> Relevant to the workflow in `simulate.py`

A demo workflow to created batched simulations for a train
and test dataset. Followed up by a merge operation.

This workflow is not executed through the `hypothesis` CLI tool, i.e., using the local backend.
As indicated below, it is possible to immediately call `python simulate.py --local` to
execute the workflow on your local machine, or on Slurm using `python simulate.py --slurm`.
As indicated in `simulate.py`, this requires some code however.

```console
you@local:~ $ python simulate.py -h
usage: simulate.py [-h] [--batch-size BATCH_SIZE] [--train TRAIN] [--test TEST] [--local]

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        Simulation batch-size (default: 10000).
  --train TRAIN         Total number of simulations for training (default: 1000000).
  --test TEST           Total number of simulations for testing (default: 100000).
  --local               Execute the workflow locally (default: false).
  --slurm               Execute the workflow on Slurm (default: false).
```

By default, the workflow will generate the necessary code to run the workflow
on a Slurm enabled HPC cluster. Importantly, it will generate a file `pipeline.bash`
which can be edited at your convenience. Executing

```console
you@local:~ $ bash pipeline.bash
```

will execute the code on the Slurm cluster with the necessary dependencies.
The status of the Slurm job can be monitored as:

```console
you@local:~ $ squeue | grep you
           1587404       all     main jhermans PD       0:00      1 (Priority)
    1587405_[0-99]       all simulate jhermans PD       0:00      1 (Dependency)
    1587406_[0-99]       all simulate jhermans PD       0:00      1 (Dependency)
           1587407       all merge_tr jhermans PD       0:00      1 (Dependency)
           1587408       all merge_te jhermans PD       0:00      1 (Dependency)
```

As you'll notice, executing this workflow for the 2nd time
is significantly shorter! This is because Hypothesis determines
what part of the computational graph needs to be computed to
ensure that the specified constraints are met.

No more recomputation and rescheduling on HPC systems!

In addition it becomes significantly easier to manage workflows on your HPC cluster,
as we keep track of the associated job identifiers. For example, executing the
following on a Slurm cluster:

```console
you@local:~ $ hypothesis workflow --name test execute simulate.py
```

Which will execute the workflow on the Slurm backend under the name `test`.
Active, and previous workflows can be retrieved using:

```console
you@local:~ $ hypothesis workflow list
```

The generated code and Slurm associated files can easily be viewed by executing
`hypothesis workflow goto <workflow name>`.

A workflow, and it's associated cluster jobs can easiliy be deleted (and cancelled) by executing

```console
you@local:~ $ hypothesis workflow delete test
```

## Example: programatically generate tasks
> Relevant to the workflow defined in `generate.py`

In some cases you might like to generate a number of tasks or jobs based on a
specified paramater to do, for instance, a parameter scan. We highlighted some
strategies in `generate.py`.
