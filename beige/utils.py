import torch
import torch.distributions as tdist

def get_alpha(expected_guide_p, size_factor, sample_mask, a0, epsilon=1e-10):
    p = expected_guide_p.permute(
                0, 2, 1) * size_factor[:, None, :]  #(n_reps, n_guides, n_bins)
    a = p/p.sum(axis=-1)[:,:,None]*a0[None,:,None]
    a=(a * sample_mask[:,None,:]).clamp(min=epsilon)
    return(a)

def get_std_normal_prob(upper_quantile: torch.Tensor, lower_quantile: torch.Tensor,
                        mu: torch.Tensor, sd: torch.Tensor,
                        mask=None) -> torch.Tensor:
    """
    Returns the probability that the normal distribution with mu and sd will
    lie between upper_quantile and lower_quantile of normal distribution
    centered at 0 and has scale sd.
    Arguments
    - mask: ignore index if 0 (=False).
    """
    inf_mask = upper_quantile == 1.0
    ninf_mask = lower_quantile == 0.0

    upper_thres = tdist.Normal(0, 1).icdf(upper_quantile)
    lower_thres = tdist.Normal(0, 1).icdf(lower_quantile)
    
    if not mask is None:
        sd = sd + (~mask).long()
    cdf_upper = tdist.Normal(mu, sd).cdf(upper_thres)
    cdf_lower = tdist.Normal(mu, sd).cdf(lower_thres)

    res = cdf_upper - cdf_lower
    res[inf_mask] = (1 - cdf_lower)[inf_mask]
    res[ninf_mask] = cdf_upper[ninf_mask]
    if not mask is None:
        res[~mask] = 0

    return(res)  

def scale_pi_by_accessibility(pi: torch.Tensor, guide_accessibility: torch.Tensor):
    """
    Arguments
    pi: reporter editing rate
    guide_accessibility: raw accessibility score
    """
    return (pi*torch.tensor(0.1280916)*torch.pow(guide_accessibility, 0.307)).clamp(max=1.0)