Module hypothesis
=================
Hypothesis is a python module for statistical inference and the
mechanisation of science.

The package contains (approximate) inference algorithms to solve inverse
statistical problems. Utilities are provided for data loading, efficient
simulation, visualization, fire-and-forget inference, and validation.

Sub-modules
-----------
* hypothesis.auto
* hypothesis.benchmark
* hypothesis.bin
* hypothesis.cli
* hypothesis.default
* hypothesis.engine
* hypothesis.exception
* hypothesis.nn
* hypothesis.plot
* hypothesis.simulation
* hypothesis.stat
* hypothesis.train
* hypothesis.util
* hypothesis.workflow

Variables
---------

    
`accelerator`
:   PyTorch device describing the accelerator backend.
    
    The variable will be initialized when ``hypothesis`` is loaded for the first
    time. It will check for the availibility of a CUDA device. If a CUDA enabled
    device is present, ``hypothesis`` will select the CUDA device defined in the
    ``CUDA_VISIBLE_DEVICES`` environment variable. If no such device is specified,
    the variable will default to GPU 0.

    
`cpu_count`
:   Number of available logical processor cores.
    
    The variable will be initialized when ``hypothesis`` is loaded for the first time.

    
`workers`
:   Default number of parallel workers in ``hypothesis``.

Functions
---------

    
`disable_gpu()`
:   Disables GPU acceleration. Hypothesis' accelerator will have been
    set to 'cpu'.

    
`enable_gpu()`
:   Tries to enable GPU acceleration. If a GPU is present, a CUDA
    device will be set, else it will default to 'cpu'.

    
`gpu_available()`
:   Checks if GPU acceleration is available.

    
`set_workers(n)`
:   Sets the number of default parallel hypothesis workers.