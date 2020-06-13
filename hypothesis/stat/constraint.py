import numpy as np



@torch.no_grad()
def highest_density_level(pdf, alpha, epsilon=10e-10):
    if pdf.ndim > 2:
        return multivariate_density_level(pdf=pdf, alpha=alpha, epsilon=epsilon)
    else:
        return univariate_density_level(pdf=pdf, alpha=alpha, epsilon=epsilon)


@torch.no_grad()
def univariate_density_level(pdf, alpha, epsilon=10e-10):
    assert(pdf.sum() == 0)
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
    raise NotImplementedError
