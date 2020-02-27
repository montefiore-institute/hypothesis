<p align="center">
    <img src="https://joerihermans.com/media/hypothesis.png" />
</p>

<p align="center">
    <img src="https://img.shields.io/badge/hypothesis-v0.0.3.ALPHA-blue.svg" />
    <img src="https://img.shields.io/badge/license-BSD-lightgrey.svg" />
</p>

A Python toolkit for (likelihood-free) inference and the mechanization of the scientific method.

## Installation

### From source

```sh
git clone https://github.com/montefiore-ai/hypothesis
cd hypothesis
pip install -e .
```

## Inference

### AALR-MCMC

TODO

### Adversarial Variational Optimization

TODO

### Amortized ratio estimation

TODO

### Approximate Bayesian Computation

TODO

### Approximate Bayesian Computation - Sequential Monte Carlo

TODO

### Likelihood-free Inference by Ratio Estimation

TODO

### Metropolis-Hastings

TODO

## Benchmark problems

### M/G/1

```python
from hypothesis.benchmark.mg1 import Simulator
from hypothesis.benchmark.mg1 import Prior

simulator = Simulator()
prior = Prior()

inputs = prior.sample((10,)) # Draw 10 samples from the prior.
outputs = simulator(inputs)
```

### Pharmacokinetic

> :heavy_check_mark: Supports experimental design

```python
from hypothesis.benchmark.pharmacokinetic import Simulator
from hypothesis.benchmark.pharmacokinetec import Prior

simulator = Simulator()
prior = Prior()

inputs = prior.sample((10,)) # Draw 10 samples from the prior.
outputs = simulator(inputs)

from hypothesis.benchmark.spatialsir import PriorExperiment # Experimental design space

prior_experiment = PriorExperiment()
experimental_designs = prior_experiment.sample((10,))

outputs = simulator(inputs, experimental_designs)
```

### SIR (Susceptible-Infected-Recovered) model

> :heavy_check_mark: Supports experimental design

```python
from hypothesis.benchmark.sir import Simulator
from hypothesis.benchmark.sir import Prior

simulator = Simulator()
prior = Prior()

inputs = prior.sample((10,)) # Draw 10 samples from the prior.
outputs = simulator(inputs)

from hypothesis.benchmark.spatialsir import PriorExperiment # Experimental design space

prior_experiment = PriorExperiment()
experimental_designs = prior_experiment.sample((10,))

outputs = simulator(inputs, experimental_designs)
```

### Spatial SIR (Susceptible-Infected-Recovered) model

> :heavy_check_mark: Supports experimental design

<p align="center">
  <img src="https://github.com/montefiore-ai/hypothesis/blob/master/.github/images/benchmark-spatialsir.gif?raw=true" />
</p>

```python
from hypothesis.benchmark.spatialsir import Simulator
from hypothesis.benchmark.spatialsir import Prior

simulator = Simulator()
prior = Prior()

inputs = prior.sample((10,)) # Draw 10 samples from the prior.
outputs = simulator(inputs)

from hypothesis.benchmark.spatialsir import PriorExperiment # Experimental design space

prior_experiment = PriorExperiment()
experimental_designs = prior_experiment.sample((10,))

outputs = simulator(inputs, experimental_designs)
```

### Tractable

```python
from hypothesis.benchmark.tractable import Simulator
from hypothesis.benchmark.tractable import Prior

simulator = Simulator()
prior = Prior()

inputs = prior.sample((10,)) # Draw 10 samples from the prior.
outputs = simulator(inputs)
```

### Weinberg

> :heavy_check_mark: Supports experimental design

```python
from hypothesis.benchmark.weinberg import Simulator
from hypothesis.benchmark.weinberg import Prior

simulator = Simulator()
prior = Prior()

inputs = prior.sample((10,)) # Draw 10 samples from the prior.
outputs = simulator(inputs)

from hypothesis.benchmark.weinberg import PriorExperiment # Experimental design space

prior_experiment = PriorExperiment()
experimental_designs = prior_experiment.sample((10,))

outputs = simulator(inputs, experimental_designs)
```

## License

Hypothesis is BSD-style licensed, as found in the LICENSE file.
