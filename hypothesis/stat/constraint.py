import numpy as np
import torch



@torch.no_grad()
def highest_density_level(pdf, alpha, epsilon=10e-7, mask=False):
    # Prepare posterior
    pdf = pdf.clone().numpy() # Clone to fix strange behaviour in Jupyter.
    total_pdf = pdf.sum()
    pdf /= total_pdf
    # Compute highest density level and the corresponding mask
    n = len(pdf)
    area = float(1)
    optimal_level = float(0)
    while area > alpha:
        m = (pdf >= optimal_level).astype(np.float32)
        area = np.sum(m * pdf)
        optimal_level += epsilon
    # Restore to original level
    optimal_level *= total_pdf

    if mask:
        return optimal_level, torch.from_numpy(m)
    else:
        return optimal_level
