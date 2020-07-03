import numpy as np
import torch


@torch.no_grad()
def highest_density_level(pdf, alpha, min_error=10e-4, mask=False, lr=0.0001, mu=0.1):
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
    last_error = error
    while abs(error) >= min_error:
        m = (pdf >= optimal_level).astype(np.float32)
        area = np.sum(m * pdf)
        error = alpha - area
        grad = lr * error
        momentum = grad + mu * momentum
        optimal_level = optimal_level - momentum
        if last_error == error:
            raise ValueError("Increase resolution of the PDF, or decrease the minimum error by supplying `min_error`.")
        last_error = error
    optimal_level *= total_pdf

    if mask:
        return optimal_level, torch.from_numpy(m)
    else:
        return optimal_level
