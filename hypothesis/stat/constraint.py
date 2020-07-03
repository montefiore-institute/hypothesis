import numpy as np
import torch



@torch.no_grad()
def highest_density_level(pdf, alpha, epsilon=10e-5, mask=False, lr=0.0001):
    # Prepare posterior
    pdf = pdf.cpu().clone().numpy() # Clone to fix strange behaviour in Jupyter.
    total_pdf = pdf.sum()
    pdf /= total_pdf
    # Compute highest density level and the corresponding mask
    n = len(pdf)
    area = float(1)
    optimal_level = float(0)
    error = float("infinity")
    momentum = 0.0
    while abs(error) >= epsilon:
        m = (pdf >= optimal_level).astype(np.float32)
        area = np.sum(m * pdf)
        error = alpha - area
        grad = lr * error
        momentum = grad + 0.9 * momentum
        optimal_level = optimal_level - momentum
    optimal_level *= total_pdf

    if mask:
        return optimal_level, torch.from_numpy(m)
    else:
        return optimal_level
