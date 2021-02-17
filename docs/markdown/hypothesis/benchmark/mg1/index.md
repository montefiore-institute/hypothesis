Module hypothesis.benchmark.mg1
===============================
The model of this benchmark describes a queuing system of continuously
arriving jobs by a single server. The time it takes to process every job is
uniformly distributed in the interval \([\\theta_1, \\theta_2]\). The arrival
between two consecutive jobs is exponentially distributed according to
the rate \(\\theta_3\). That is, for
every job \(i\) we have the processing time \(p_i\) , an arrival
time \(a_i\) and the time \(l_i\) at which the job left the queue.

Sub-modules
-----------
* hypothesis.benchmark.mg1.simulator
* hypothesis.benchmark.mg1.util