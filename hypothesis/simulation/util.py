import torch



def sample_joint(simulator, prior, n=1):
    r""""""
    inputs = prior.sample(torch.Size([n])).view(n, -1)
    outputs = simulator(inputs)

    return inputs, outputs



def joint_sampler(simulator, prior):
    r""""""
    yield sample_joint(simulator, prior, n=1)



def sample_marginal(simulator, prior, n=1):
    r""""""
    _, outputs = sample_joint(simulator, prior, n=n)

    return outputs



def marginal_sampler(simulator, prior):
    r""""""
    yield sample_marginal(simulator, prior, n=1)



def sample_likelihood(simulator, input, n=1):
    r""""""
    inputs = input.view(1, -1).repeat(n, 1)
    outputs = simulator(inputs)

    return outputs



def likelihood_sampler(simulator, input):
    r""""""
    yield sample_likelihood(simulator, input, n=1)
