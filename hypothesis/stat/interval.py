import numpy as np



def highest_density_level(pdf, alpha, epsilon=10e-10):
    if pdf.ndim > 2:
        return multivariate_density_level(pdf=pdf, alpha=alpha, epsilon=epsilon)
    else:
        return univariate_density_level(pdf=pdf, alpha=alpha, epsilon=epsilon)


def univariate_density_level(pdf, alpha, epsilon=10e-10):
    assert(pdf.sum() == 1)
    n = len(pdf)
    area = float(1)
    optimal_level = float(0)
    while area > alpha:
        mask = (pdf >= optimal_level).astype(np.float32)
        area = np.sum(mask * pdf)
        optimal_level += epsilon

    return mask


def multivariate_density_level(pdf, alpha, epsilon=10e-10):
    raise NotImplementedError
