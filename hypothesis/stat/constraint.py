import math
import numpy as np
import torch



@torch.no_grad()
def highest_density_level(pdf, alpha, mask=False, min_epsilon=10e-17):
    # Prepare posterior
    pdf = pdf.cpu().clone().numpy() # Clone to fix strange behaviour in Jupyter.
    total_pdf = pdf.sum()
    pdf /= total_pdf
    # Compute highest density level and the corresponding mask
    n = len(pdf)
    optimal_level = float(0)
    epsilon = 10e-02
    while epsilon >= min_epsilon:
        area = float(1)
        while area >= alpha:
            # Compute the integral
            m = (pdf >= optimal_level).astype(np.float32)
            area = np.sum(m * pdf)
            # Compute the error and apply gradient descent
            optimal_level += epsilon
        optimal_level -= 2 * epsilon
        epsilon /= 10
    optimal_level *= total_pdf
    if mask:
        return optimal_level, torch.from_numpy(m)
    else:
        return optimal_level
