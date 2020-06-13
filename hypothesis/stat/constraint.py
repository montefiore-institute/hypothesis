import numpy as np
import torch



@torch.no_grad()
def highest_density_level(pdf, alpha, epsilon=10e-10, mask=False):
    # Prepare posterior
    total_pdf = pdf.sum().item()
    pdf /= total_pdf
    pdf = pdf.numpy()
    # Compute highest density level and the corresponding mask
    n = len(pdf)
    area = float(1)
    optimal_level = float(0)
    while area > alpha:
        mask = (pdf >= optimal_level).astype(np.float32)
        area = np.sum(mask * pdf)
        optimal_level += epsilon
    # Restore to original level
    optimal_level *= total_pdf

    if mask
        return optimal_level, torch.from_numpy(mask)
    else:
        return optimal_level
