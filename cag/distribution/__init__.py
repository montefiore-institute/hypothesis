"""
Distributions API.
"""

import torch

from torch.distributions.bernoulli import Bernoulli
from torch.distributions.beta import Beta
from torch.distributions.binomial import Binomial
from torch.distributions.categorical import Categorical
from torch.distributions.cauchy import Cauchy
from torch.distributions.chi2 import Chi2
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.exponential import Exponential
from torch.distributions.fishersnedecor import FisherSnedecor
from torch.distributions.gamma import Gamma
from torch.distributions.geometric import Geometric
from torch.distributions.gumbel import Gumbel
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.half_normal import HalfNormal
from torch.distributions.independent import Independent
from torch.distributions.laplace import Laplace
from torch.distributions.log_normal import LogNormal
from torch.distributions.logistic_normal import LogisticNormal
from torch.distributions.multinomial import Multinomial
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.pareto import Pareto
from torch.distributions.poisson import Poisson
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.studentT import StudentT
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.uniform import Uniform
from torch.distributions.weibull import Weibull



def show_torch_distributions():
    """Import the PyTorch's `distributions` API.

    Please don't kill me.
    """
    import pkgutil
    package = torch.distributions
    for importer, module_name, is_module in pkgutil.walk_packages(package.__path__):
        words = [word[0].capitalize() + word[1:] for word in module_name.split('_')]
        distribution_name = "".join(words)
        print("from torch.distributions." + module_name + " import " + distribution_name)
