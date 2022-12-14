import numpy as np
import torch
from scipy.optimize import curve_fit

"""
Returns fitted alpha 0 (sum of concentration) of Dirichlet-Multinomial
For the notation, see DESeq paper
"""

def get_valid_idx(x:torch.Tensor, y:torch.Tensor):
    return ~(torch.isnan(x) | torch.isnan(y) | torch.isinf(x) | torch.isinf(y))

def get_valid_vals(x, y):
    valid = get_valid_idx(x, y)
    return(x[valid].detach().numpy(), y[valid].detach().numpy())


def get_q(X, sample_size_factors, sample_mask = None):
    """
    Obtain depth-normalized sample mean
    """
    if sample_mask is None:
        sample_mask = torch.ones((n_reps, n_condits))
    n_reps, n_condits, n_guides = X.shape
    q_condit_guide = ((X/sample_size_factors[:,:,None])*sample_mask[:,:,None]).sum(axis=0)/sample_mask.sum(axis=0)[:,None]
    assert q_condit_guide.shape == (n_condits, n_guides)
    return(q_condit_guide)

def get_w(X, sample_size_factors, sample_mask = None):
    """
    Obtain depth-normalized sample variance
    """
    n_reps, n_condits, n_guides = X.shape
    if sample_mask is None:
        sample_mask = torch.ones((n_reps, n_condits))
    q = get_q(X, sample_size_factors, sample_mask)
    se = (X/sample_size_factors[:,:,None] - q)**2
    w=(se * sample_mask[:,:,None]).sum(axis=0)/(sample_mask.sum(axis=0))[:,None]
    return(w)

def linear(x, b0, b1):
    return b0 + b1*x

def get_fitted_alpha0(X, sample_size_factors, sample_mask = None):
    n_reps, n_condits, n_guides = X.shape
    if sample_mask is None:
        sample_mask = torch.ones((n_reps, n_condits))
    w = get_w(X, sample_size_factors, sample_mask = sample_mask)
    q = get_q(X, sample_size_factors, sample_mask = sample_mask)
    n = q.sum(axis=0)   # depth-normalized total counts across bins
    p = q/n[None,:]     # depth-normalized p for Multinomial
    multinom_var = n[None,:]*p*(1-p)    # theoretical Multinomial variance
    r = w/multinom_var
    a0 = (n/(r - 1 + 1/(1-p))).mean(axis=0)

    x, y = get_valid_vals(n.log(), a0.log())
    popt, pcov = curve_fit(linear, x, y)
    print("Linear fit of log(a0) ~ log(q): [b0, b1]={}, cov={}".format(popt, pcov))
    a0_est = np.exp(linear(q.sum(axis=0).log().numpy(), *popt))
    return(a0_est)
