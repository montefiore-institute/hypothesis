import numpy as np
import torch



@torch.no_grad()
def highest_density_level(pdf, alpha, epsilon=10e-10):
    assert(pdf.sum() == 1)

    pdf = pdf.numpy()
    if pdf.ndim >= 2:
        mask = multivariate_density_level(pdf=pdf, alpha=alpha, epsilon=epsilon)
    else:
        mask = univariate_density_level(pdf=pdf, alpha=alpha, epsilon=epsilon)

    return torch.from_numpy(mask)


@torch.no_grad()
def univariate_density_level(pdf, alpha, epsilon=10e-10):
    n = len(pdf)
    area = float(1)
    optimal_level = float(0)
    while area > alpha:
        mask = (pdf >= optimal_level).astype(np.float32)
        area = np.sum(mask * pdf)
        optimal_level += epsilon

    return mask


@torch.no_grad()
def multivariate_density_level(pdf, alpha, epsilon=10e-10):
    n = len(pdf)
    area = float(1)
    optimal_level = float(0)
    while area > alpha:
        mask = (pdf >= optimal_level).astype(np.float32)
        area = np.sum(mask * pdf)
        optimal_level += epsilon

    return mask
