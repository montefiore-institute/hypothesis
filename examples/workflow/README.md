# Workflows

TODO

## Usage

### Execute the workflow directly

TODO

### Using the `hypothesis` CLI tool

TODO

## Example: simulation pipeline
A demo workflow to created batched simulations for a train
and test dataset. Followed up by a merge operation.

This workflow is not executed through the `hypothesis` CLI tool.

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
what part of the computational graph need to be computed to
ensure that the specified constraints are met.

No more recomputation and rescheduling on HPC systems!
