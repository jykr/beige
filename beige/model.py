import torch
import torch.distributions as tdist
import pyro
from pyro import poutine
import pyro.distributions as dist
import torch.distributions.constraints as constraints


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

def NormalModel(data, mask_thres = 10, use_bcmatch = True):
    '''
    Fit only on guide counts
    '''
    replicate_plate = pyro.plate("rep_plate", data.n_reps, dim=-3)
    replicate_plate2 = pyro.plate("rep_plate2", data.n_reps, dim=-2)
    bin_plate = pyro.plate("bin_plate", data.n_bins, dim=-2)
    guide_plate = pyro.plate("guide_plate", data.n_guides, dim=-1)

    # Set the prior for phenotype means
    with pyro.plate('guide_plate0', 1):
        with pyro.plate('guide_plate1', data.n_targets):
            mu_alleles = pyro.sample('mu_alleles', dist.Laplace(0, 1))
            sd_alleles = pyro.sample("sd_alleles",  dist.LogNormal(
                torch.zeros((data.n_targets, 1)), torch.ones(data.n_targets, 1)))
    mu_center = mu_alleles
    mu = torch.repeat_interleave(
        mu_center, data.target_lengths, dim=0)
    assert mu.shape == (data.n_guides, 1)
    sd = sd_alleles
    sd = torch.repeat_interleave(sd, data.target_lengths, dim=0)
    assert sd.shape == (data.n_guides, 1)

    with replicate_plate as r:
        with bin_plate as b:
            uq = data.upper_bounds[b]
            lq = data.lower_bounds[b]
            assert uq.shape == lq.shape == (data.n_bins,)
            # with guide_plate, poutine.mask(mask=(data.allele_counts.sum(axis=-1) == 0)):
            with guide_plate:
                alleles_p_bin = get_std_normal_prob(
                    uq.unsqueeze(-1).unsqueeze(-1).expand((-1,
                                                           data.n_guides, 1)),
                    lq.unsqueeze(-1).unsqueeze(-1).expand((-1,
                                                           data.n_guides, 1)),
                    mu.unsqueeze(0).expand((data.n_bins, -1, -1)),
                    sd.unsqueeze(0).expand((data.n_bins, -1, -1)))
                assert alleles_p_bin.shape == (data.n_bins, data.n_guides, 1)

            expected_allele_p = alleles_p_bin.unsqueeze(
                0).expand(data.n_reps, -1, -1, -1)
            expected_guide_p = expected_allele_p.sum(axis=-1)
            assert expected_guide_p.shape == (
                data.n_reps, data.n_bins, data.n_guides), expected_guide_p.shape

    with replicate_plate2:
        with pyro.plate("guide_plate3", data.n_guides, dim=-1):
            a = get_alpha(expected_guide_p, data.size_factor, data.sample_mask, data.a0)
            a_bcmatch = a
            a_bcmatch = a
            #a_bcmatch = get_alpha(expected_guide_p, data.size_factor_bcmatch, data.sample_mask, data.a0_bcmatch)
            #assert a.shape == a_bcmatch.shape == (data.n_reps, data.n_guides, data.n_bins)
            assert data.X.shape == data.X_bcmatch.shape == (
                data.n_reps, data.n_bins, data.n_guides,)
            with poutine.mask(mask=torch.logical_and(data.X.permute(0, 2, 1).sum(axis=-1) > mask_thres, data.repguide_mask)):
                pyro.sample("guide_counts",
                            dist.DirichletMultinomial(
                                a, validate_args=False),
                            obs = data.X_masked.permute(0, 2, 1))
            if use_bcmatch:
                with poutine.mask(mask=torch.logical_and(data.X_bcmatch.permute(0, 2, 1).sum(axis=-1) > mask_thres, data.repguide_mask)):
                    pyro.sample("guide_bcmatch_counts",
                                dist.DirichletMultinomial(
                                    a_bcmatch, validate_args=False),
                                obs = data.X_bcmatch_masked.permute(0, 2, 1))

    return(alleles_p_bin)

def MixtureNormalFittedPi(data, alpha_prior=1, use_bcmatch = True):
    '''
    Fit only on guide counts. Estimate alpha
    '''
    replicate_plate = pyro.plate("rep_plate", data.n_reps, dim=-3)
    replicate_plate2 = pyro.plate("rep_plate2", data.n_reps, dim=-2)
    bin_plate = pyro.plate("bin_plate", data.n_bins, dim=-2)
    guide_plate = pyro.plate("guide_plate", data.n_guides, dim=-1)

    # Set the prior for phenotype means
    with pyro.plate('guide_plate0', 1):
        with pyro.plate('guide_plate1', data.n_targets):
            mu_alleles = pyro.sample('mu_alleles', dist.Laplace(0, 1))
            sd_alleles = pyro.sample("sd_alleles",  dist.LogNormal(
                torch.zeros((data.n_targets, 1)), torch.ones(data.n_targets, 1)))
    mu_center = torch.cat(
        [mu_alleles, torch.zeros((data.n_targets, 1))], axis=-1)
    mu = torch.repeat_interleave(
        mu_center, data.target_lengths, dim=0)
    assert mu.shape == (data.n_guides, 2)

    sd = torch.cat([sd_alleles, torch.ones((data.n_targets, 1))], axis=-1)
    sd = torch.repeat_interleave(sd, data.target_lengths, dim=0)
    assert sd.shape == (data.n_guides, 2)
    # The pi should be Dirichlet distributed instead of independent betas
    alpha_pi = pyro.param("alpha_pi", torch.ones(
        (data.n_guides, 2,))*alpha_prior, constraint=constraints.positive)
    assert alpha_pi.shape == (data.n_guides, 2,), alpha_pi.shape

    with replicate_plate as r:
        with bin_plate as b:
            uq = data.upper_bounds[b]
            lq = data.lower_bounds[b]
            assert uq.shape == lq.shape == (data.n_bins,)
            # with guide_plate, poutine.mask(mask=(data.allele_counts.sum(axis=-1) == 0)):
            with guide_plate:
                alleles_p_bin = get_std_normal_prob(
                    uq.unsqueeze(-1).unsqueeze(-1).expand((-1,
                                                           data.n_guides, 2)),
                    lq.unsqueeze(-1).unsqueeze(-1).expand((-1,
                                                           data.n_guides, 2)),
                    mu.unsqueeze(0).expand((data.n_bins, -1, -1)),
                    sd.unsqueeze(0).expand((data.n_bins, -1, -1)))
                assert alleles_p_bin.shape == (data.n_bins, data.n_guides, 2)

            expected_allele_p = alpha_pi.unsqueeze(0).unsqueeze(0).expand(
                data.n_reps, data.n_bins, -1, -1) * alleles_p_bin[None, :, :, :]
            expected_guide_p = expected_allele_p.sum(axis=-1)
            assert expected_guide_p.shape == (
                data.n_reps, data.n_bins, data.n_guides), expected_guide_p.shape

    with replicate_plate2:
        with pyro.plate("guide_plate3", data.n_guides, dim=-1):
            a = get_alpha(expected_guide_p, data.size_factor, data.sample_mask, data.a0)
            a_bcmatch = a
            #a_bcmatch = get_alpha(expected_guide_p, data.size_factor_bcmatch, data.sample_mask, data.a0_bcmatch)
            #assert a.shape == a_bcmatch.shape == (data.n_reps, data.n_guides, data.n_bins)
            assert data.X.shape == data.X_bcmatch.shape == (
                data.n_reps, data.n_bins, data.n_guides,)
            with poutine.mask(mask=torch.logical_and(data.X.permute(0, 2, 1).sum(axis=-1) > 10, data.repguide_mask)):
                pyro.sample("guide_counts",
                            dist.DirichletMultinomial(
                                a, validate_args=False),
                            obs = data.X_masked.permute(0, 2, 1))
            if use_bcmatch:
                with poutine.mask(mask=torch.logical_and(data.X_bcmatch.permute(0, 2, 1).sum(axis=-1) > 10, data.repguide_mask)):
                    pyro.sample("guide_bcmatch_counts",
                                dist.DirichletMultinomial(
                                    a_bcmatch, validate_args=False),
                                obs = data.X_bcmatch_masked.permute(0, 2, 1))
    return(alleles_p_bin)

def MixtureNormal(data, alpha_prior=1, use_bcmatch = True, sd_scale=0.01, norm_pi=False):
    '''
    model B + proper pi
    '''
    if data.sorting_scheme == "topbot":
        replicate_plate = pyro.plate("topbot_plate", data.n_reps, dim=-3)
        replicate_plate2 = pyro.plate("topbot_plate2", data.n_reps, dim=-2)
        bin_plate = pyro.plate("tb_bin_plate", data.n_bins, dim=-2)
    else:
        replicate_plate = pyro.plate("rep_plate", data.n_reps, dim=-3)
        replicate_plate2 = pyro.plate("rep_plate2", data.n_reps, dim=-2)
        bin_plate = pyro.plate("bin_plate", data.n_bins, dim=-2)
    guide_plate = pyro.plate("guide_plate", data.n_guides, dim=-1)

    # Set the prior for phenotype means
    with pyro.plate('guide_plate0', 1):
        with pyro.plate('guide_plate1', data.n_targets):
            mu_alleles = pyro.sample('mu_alleles', dist.Laplace(0, 1))
            sd_alleles = pyro.sample("sd_alleles",  dist.LogNormal(
                torch.zeros((data.n_targets, 1)), torch.ones(data.n_targets, 1)*sd_scale))
    mu_center = torch.cat(
        [mu_alleles, torch.zeros((data.n_targets, 1))], axis=-1)
    mu = torch.repeat_interleave(
        mu_center, data.target_lengths, dim=0)
    assert mu.shape == (data.n_guides, 2)

    sd = torch.cat([sd_alleles, torch.ones((data.n_targets, 1))], axis=-1)
    sd = torch.repeat_interleave(sd, data.target_lengths, dim=0)
    assert sd.shape == (data.n_guides, 2)
    # The pi should be Dirichlet distributed instead of independent betas
    alpha_pi = pyro.param("alpha_pi", torch.ones(
        (data.n_guides, 2,))*alpha_prior, constraint=constraints.positive)
    pi_a_scaled = alpha_pi/alpha_pi.sum(axis=-1)[:,None]*data.pi_a0[:,None]
    assert alpha_pi.shape == (data.n_guides, 2,), alpha_pi.shape
    with replicate_plate:
        with guide_plate:
            # Accounting for sample specific overall edit rate across all guides.
            # P(allele | guide, bin=bulk)
            pi = pyro.sample(
                "pi", 
                dist.Dirichlet(pi_a_scaled.unsqueeze(0).unsqueeze(0).expand(data.n_reps, 1, -1, -1)))
            assert pi.shape == (data.n_reps, 1, data.n_guides, 2,), pi.shape
            pyro.sample(
                "bulk_allele_count",
                dist.Multinomial(probs=pi.unsqueeze(
                    0).unsqueeze(1), validate_args=False),
                obs=data.allele_counts_bulk
            )
    with replicate_plate as r:  
        with bin_plate as b:
            uq = data.upper_bounds[b]
            lq = data.lower_bounds[b]
            assert uq.shape == lq.shape == (data.n_bins,)
            # with guide_plate, poutine.mask(mask=(data.allele_counts.sum(axis=-1) == 0)):
            with guide_plate, poutine.mask(mask=data.repguide_mask.unsqueeze(1)):
                alleles_p_bin = get_std_normal_prob(
                    uq.unsqueeze(-1).unsqueeze(-1).expand((-1,
                                                           data.n_guides, 2)),
                    lq.unsqueeze(-1).unsqueeze(-1).expand((-1,
                                                           data.n_guides, 2)),
                    mu.unsqueeze(0).expand((data.n_bins, -1, -1)),
                    sd.unsqueeze(0).expand((data.n_bins, -1, -1)))
                assert alleles_p_bin.shape == (data.n_bins, data.n_guides, 2)

            expected_allele_p = pi.expand(data.n_reps, data.n_bins, -1, -1) * alleles_p_bin[None, :, :, :]
            expected_guide_p = expected_allele_p.sum(axis=-1)
            assert expected_guide_p.shape == (
                data.n_reps, data.n_bins, data.n_guides), expected_guide_p.shape

    with replicate_plate2:
        with pyro.plate("guide_plate3", data.n_guides, dim=-1):
            a = get_alpha(expected_guide_p, data.size_factor, data.sample_mask, data.a0)
            a_bcmatch = a
            #a_bcmatch = get_alpha(expected_guide_p, data.size_factor_bcmatch, data.sample_mask, data.a0_bcmatch)
            #assert a.shape == a_bcmatch.shape == (data.n_reps, data.n_guides, data.n_bins)
            assert data.X.shape == data.X_bcmatch.shape == (
                data.n_reps, data.n_bins, data.n_guides,)
            with poutine.mask(mask=torch.logical_and(data.X.permute(0, 2, 1).sum(axis=-1) > 10, data.repguide_mask)):
                pyro.sample("guide_counts",
                            dist.DirichletMultinomial(
                                a, validate_args=False),
                            obs = data.X_masked.permute(0, 2, 1))
            if use_bcmatch:
                with poutine.mask(mask=torch.logical_and(data.X_bcmatch.permute(0, 2, 1).sum(axis=-1) > 10, data.repguide_mask)):
                    pyro.sample("guide_bcmatch_counts",
                                dist.DirichletMultinomial(
                                    a_bcmatch, validate_args=False),
                                obs = data.X_bcmatch_masked.permute(0, 2, 1))

def guide_MixtureNormal(data, alpha_prior=1):
    '''
    Guide for MixtureNormal model
    '''
    if data.sorting_scheme == "topbot":
        replicate_plate = pyro.plate("topbot_plate", data.n_reps, dim=-3)
        replicate_plate2 = pyro.plate("topbot_plate2", data.n_reps, dim=-2)
        bin_plate = pyro.plate("tb_bin_plate", data.n_bins, dim=-2)
    else:
        replicate_plate = pyro.plate("rep_plate", data.n_reps, dim=-3)
        replicate_plate2 = pyro.plate("rep_plate2", data.n_reps, dim=-2)
        bin_plate = pyro.plate("bin_plate", data.n_bins, dim=-2)
    guide_plate = pyro.plate("guide_plate", data.n_guides, dim=-1)

    # Set the prior for phenotype means
    mu_loc = pyro.param("mu_loc", torch.zeros((data.n_targets, 1)))
    mu_scale = pyro.param("mu_scale", torch.ones((data.n_targets, 1)), constraint = constraints.positive)
    sd_loc = pyro.param("sd_loc", torch.zeros((data.n_targets, 1)))
    sd_scale = pyro.param("sd_scale", torch.ones((data.n_targets, 1)), constraint = constraints.positive)
    with pyro.plate('guide_plate0', 1):
        with pyro.plate('guide_plate1', data.n_targets):
            mu_alleles = pyro.sample('mu_alleles', dist.Normal(mu_loc, mu_scale))
            sd_alleles = pyro.sample("sd_alleles",  dist.LogNormal(sd_loc, sd_scale))
    mu_center = torch.cat(
        [mu_alleles, torch.zeros((data.n_targets, 1))], axis=-1)
    mu = torch.repeat_interleave(
        mu_center, data.target_lengths, dim=0)
    assert mu.shape == (data.n_guides, 2)

    sd = torch.cat([sd_alleles, torch.ones((data.n_targets, 1))], axis=-1)
    sd = torch.repeat_interleave(sd, data.target_lengths, dim=0)
    assert sd.shape == (data.n_guides, 2)
    # The pi should be Dirichlet distributed instead of independent betas
    alpha_pi = pyro.param("alpha_pi", torch.ones(
        (data.n_guides, 2,))*alpha_prior, constraint=constraints.positive)
    assert alpha_pi.shape == (data.n_guides, 2,), alpha_pi.shape
    pi_a_scaled = alpha_pi/alpha_pi.sum(axis=-1)[:,None]*data.pi_a0[:,None]
    with replicate_plate:
        with guide_plate:
            pi = pyro.sample(
                "pi", 
                dist.Dirichlet(pi_a_scaled.unsqueeze(0).unsqueeze(0).expand(data.n_reps, 1, -1, -1).clamp(1e-5)))
            assert pi.shape == (data.n_reps, 1, data.n_guides, 2,), pi.shape


def MixtureNormalRepPi(data, alpha_prior=1, scale_alpha = True, use_bcmatch = True):
    '''
    Fit only on guide counts
    '''
    replicate_plate = pyro.plate("rep_plate", data.n_reps, dim=-3)
    replicate_plate2 = pyro.plate("rep_plate2", data.n_reps, dim=-2)
    bin_plate = pyro.plate("bin_plate", data.n_bins, dim=-2)
    guide_plate = pyro.plate("guide_plate", data.n_guides, dim=-1)

    # Set the prior for phenotype means
    with pyro.plate('guide_plate0', 1):
        with pyro.plate('guide_plate1', data.n_targets):
            mu_alleles = pyro.sample('mu_alleles', dist.Laplace(0, 1))
            sd_alleles = pyro.sample("sd_alleles",  dist.LogNormal(
                torch.zeros((data.n_targets, 1)), torch.ones(data.n_targets, 1)))
    mu_center = torch.cat(
        [mu_alleles, torch.zeros((data.n_targets, 1))], axis=-1)
    mu = torch.repeat_interleave(
        mu_center, data.target_lengths, dim=0)
    assert mu.shape == (data.n_guides, 2)

    sd = torch.cat([sd_alleles, torch.ones((data.n_targets, 1))], axis=-1)
    sd = torch.repeat_interleave(sd, data.target_lengths, dim=0)
    assert sd.shape == (data.n_guides, 2)
    # The pi should be Dirichlet distributed instead of independent betas
    alpha_pi= pyro.param("alpha_pi", torch.ones(
        (data.n_guides, 2,))*alpha_prior, constraint=constraints.positive)

    # Account for per sample variation of overall edit rate.
    # This will be multiplied to the edit rate of the positive alleles (with any edits).
    replicate_edit_scale = pyro.param("replicate_edit_scale",
                                      torch.ones(data.n_total_reps,),
                                      constraint=constraints.positive)
    if data.sorting_scheme == "topbot":
        replicate_edit_scale_per_allele = torch.cat(
            (replicate_edit_scale[:data.n_reps].unsqueeze(-1).expand(-1, 2-1),  # (n_reps, 1)
            torch.ones((data.n_reps, 1))),
            axis=-1)
    else:
        replicate_edit_scale_per_allele = torch.cat(
            (replicate_edit_scale[(data.n_total_reps - data.n_reps):].unsqueeze(-1).expand(-1, 2-1),  # (n_reps, 1)
            torch.ones((data.n_reps, 1))),
            axis=-1)

    assert replicate_edit_scale_per_allele.shape == (
        data.n_reps, 2,), replicate_edit_scale_per_allele.shape

    replicate_alpha_pi = alpha_pi[None, :, :] * \
        replicate_edit_scale_per_allele[:, None, :]
    assert replicate_alpha_pi.shape == (data.n_reps, data.n_guides, 2,)

    assert alpha_pi.shape == (data.n_guides, 2,), alpha_pi.shape
    with replicate_plate:
        with guide_plate:
            pyro.sample(
                "bulk_allele_count",
                dist.DirichletMultinomial(
                    replicate_alpha_pi.unsqueeze(1), validate_args=False),
                obs=data.allele_counts_bulk
            )

    with replicate_plate as r:
        with bin_plate as b:
            uq = data.upper_bounds[b]
            lq = data.lower_bounds[b]
            assert uq.shape == lq.shape == (data.n_bins,)
            # with guide_plate, poutine.mask(mask=(data.allele_counts.sum(axis=-1) == 0)):
            with guide_plate:
                alleles_p_bin = get_std_normal_prob(
                    uq.unsqueeze(-1).unsqueeze(-1).expand((-1,
                                                           data.n_guides, 2)),
                    lq.unsqueeze(-1).unsqueeze(-1).expand((-1,
                                                           data.n_guides, 2)),
                    mu.unsqueeze(0).expand((data.n_bins, -1, -1)),
                    sd.unsqueeze(0).expand((data.n_bins, -1, -1)))
                assert alleles_p_bin.shape == (data.n_bins, data.n_guides, 2)

            expected_allele_p = replicate_alpha_pi.unsqueeze(1).expand(
                -1, data.n_bins, -1, -1) * alleles_p_bin[None, :, :, :]
            expected_guide_p = expected_allele_p.sum(axis=-1)
            assert expected_guide_p.shape == (
                data.n_reps, data.n_bins, data.n_guides), expected_guide_p.shape
    if scale_alpha:
        alpha_scaling_factor = pyro.param('alpha_scaling_factor', torch.tensor(1.0),
    constraint=constraints.positive)
    else:
        alpha_scaling_factor = 1.0
    with replicate_plate2:
        with pyro.plate("guide_plate3", data.n_guides, dim=-1):
            a = get_alpha(expected_guide_p, data.size_factor, data.sample_mask, data.a0)
            a_bcmatch = a
            #a_bcmatch = get_alpha(expected_guide_p, data.size_factor_bcmatch, data.sample_mask, data.a0_bcmatch)
            #assert a.shape == a_bcmatch.shape == (data.n_reps, data.n_guides, data.n_bins)
            assert data.X.shape == data.X_bcmatch.shape == (
                data.n_reps, data.n_bins, data.n_guides,)
            with poutine.mask(mask=torch.logical_and(data.X.permute(0, 2, 1).sum(axis=-1) > 10, data.repguide_mask)):
                pyro.sample("guide_counts",
                            dist.DirichletMultinomial(
                                a*alpha_scaling_factor, validate_args=False),
                            obs = data.X_masked.permute(0, 2, 1))
            if use_bcmatch:
                with poutine.mask(mask=torch.logical_and(data.X_bcmatch.permute(0, 2, 1).sum(axis=-1) > 10, data.repguide_mask)):
                    pyro.sample("guide_bcmatch_counts",
                                dist.DirichletMultinomial(
                                    a_bcmatch*alpha_scaling_factor, validate_args=False),
                                obs = data.X_bcmatch_masked.permute(0, 2, 1))

    return(alleles_p_bin)

def MulticomponentMixtureNormal(data, alpha_prior=1, scale_alpha = True, use_bcmatch = True):
    '''
    Fit only on guide counts
    '''
    replicate_plate = pyro.plate("rep_plate", data.n_reps, dim=-3)
    replicate_plate2 = pyro.plate("rep_plate2", data.n_reps, dim=-2)
    bin_plate = pyro.plate("bin_plate", data.n_bins, dim=-2)
    guide_plate = pyro.plate("guide_plate", data.n_guides, dim=-1)

    # Set the prior for phenotype means
    with pyro.plate('guide_plate1', data.n_edits):
        mu_edits = pyro.sample('mu_alleles', dist.Laplace(0, 1))
        sd_edits = pyro.sample("sd_alleles",  dist.LogNormal(
            torch.zeros((data.n_edits,)), torch.ones(data.n_edits,)))
    assert mu_edits.shape == sd_edits.shape == (data.n_edits,)
    assert data.allele_to_edit.shape == (data.n_guides, data.n_max_alleles-1, data.n_edits)
    mu_alleles = torch.matmul(data.allele_to_edit, mu_edits)
    assert mu_alleles.shape == (data.n_guides, data.n_max_alleles-1)
    sd_alleles = torch.linalg.norm(data.allele_to_edit * sd_edits[None,None,:], dim=-1) #Frobenius 2-norm
    
    mu = torch.cat([torch.zeros((data.n_guides, 1)), mu_alleles], axis=-1)
    sd = torch.cat([torch.ones((data.n_guides, 1)), sd_alleles], axis=-1)
    assert mu.shape == sd.shape == (
        data.n_guides, data.n_max_alleles), (mu.shape, sd.shape)

    # The pi should be Dirichlet distributed instead of independent betas
    alpha_pi0 = torch.ones((data.n_guides, data.n_max_alleles,))*alpha_prior
    # Effectively remove alphas for non-existing alleles
    alpha_pi0[~data.allele_mask] = 1e-10
    alpha_pi= pyro.param("alpha_pi", alpha_pi0, constraint=constraints.positive)
    alpha_pi[~data.allele_mask] = 1e-10
    
    with replicate_plate:
        with guide_plate:
            pyro.sample(
                "bulk_allele_count",
                dist.DirichletMultinomial(alpha_pi.unsqueeze(
                    0).unsqueeze(1), validate_args=False),
                obs=data.allele_counts_bulk
            )

    with replicate_plate as r:
        with bin_plate as b:
            uq = data.upper_bounds[b]
            lq = data.lower_bounds[b]
            assert uq.shape == lq.shape == (data.n_bins,)
            with guide_plate:
                alleles_p_bin = get_std_normal_prob(
                    uq.unsqueeze(-1).unsqueeze(-1).expand((-1,
                                                           data.n_guides, data.n_max_alleles)),
                    lq.unsqueeze(-1).unsqueeze(-1).expand((-1,
                                                           data.n_guides, data.n_max_alleles)),
                    mu.unsqueeze(0).expand((data.n_bins, -1, -1)),
                    sd.unsqueeze(0).expand((data.n_bins, -1, -1)),
                    mask=data.allele_mask.unsqueeze(0).expand((data.n_bins, -1, -1)))
                assert alleles_p_bin.shape == (
                    data.n_bins, data.n_guides, data.n_max_alleles)
            expected_allele_p = alpha_pi.unsqueeze(0).unsqueeze(0).expand(
                data.n_reps, data.n_bins, -1, -1) * alleles_p_bin[None, :, :, :]
            expected_guide_p = expected_allele_p.sum(axis=-1)
            assert expected_guide_p.shape == (
                data.n_reps, data.n_bins, data.n_guides), expected_guide_p.shape

    with replicate_plate2:
        with guide_plate:
            a = get_alpha(expected_guide_p, data.size_factor, data.sample_mask, data.a0)
            a_bcmatch = a
            #a_bcmatch = get_alpha(expected_guide_p, data.size_factor_bcmatch, data.sample_mask, data.a0_bcmatch)
            #assert a.shape == a_bcmatch.shape == (data.n_reps, data.n_guides, data.n_bins)
            assert data.X.shape == data.X_bcmatch.shape == (
                data.n_reps, data.n_bins, data.n_guides,)
            with poutine.mask(mask=torch.logical_and(data.X.permute(0, 2, 1).sum(axis=-1) > 10, data.repguide_mask)):
                
                pyro.sample("guide_counts",
                            dist.DirichletMultinomial(
                                a, validate_args=False),
                            obs = data.X_masked.permute(0, 2, 1))
            if use_bcmatch:
                with poutine.mask(mask=torch.logical_and(data.X_bcmatch.permute(0, 2, 1).sum(axis=-1) > 10, data.repguide_mask)):
                    pyro.sample("guide_bcmatch_counts",
                                dist.DirichletMultinomial(
                                    a_bcmatch, validate_args=False),
                                obs = data.X_bcmatch_masked.permute(0, 2, 1))
