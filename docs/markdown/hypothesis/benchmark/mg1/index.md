Module hypothesis.benchmark.mg1
===============================
Simulation model of the M/G/1 queuing model.

This model describes a queuing system of continuously arriving jobs by a
single server. The time it takes to process every job is uniformly
distributed in the interval :math:`[\theta_1, \theta_2]`. The arrival
between two consecutive jobs is exponentially distributed according to
the rate :math:`\theta_3`. That is, for
every job :math:`i` we have the processing time :math:`p_i` , an arrival
time :math:`a_i` and the time :math:`l_i` at which the job left the queue.

Sub-modules
-----------
* hypothesis.benchmark.mg1.simulator
* hypothesis.benchmark.mg1.util