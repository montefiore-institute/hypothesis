# Project example

This example highlights how we setup a posterior estimation project with Hypothesis.
It demonstrates various aspects like how to use the predefined trainer, attach hooks
to the optimizer, monitor using tensorboard etc.

The training procedure can be started using
```console
you@local:~ $ sh train.sh
```

The progress can be monitored through the terminal output (since `--show` has been specified).
To monitor the progress in TensorBoard, start the TensorBoard instance
```console
you@local:~ $ tensorboard --logdir tensorboard
```
