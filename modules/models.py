from lncass import (sample_lncass_single_psr,
                    PRIOR_DICT,
                    sample_lncass_pta,
                    sample_lncass_single_psr_javier_method,
                    sample_lncass_single_psr_alt)
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import jax
import numpy as np
import discovery as ds
import scipy.linalg as sl
from scipy.special import logsumexp
import jax.debug
from interpax import interp1d
import transformations
from jax.scipy.integrate import trapezoid

def create_amatrix(npoints, logf, cond=0):
    """Create inverse of A matrix for inverting linear interpolation.
    Also return extra pieces needed to make interpolation correct at the edges.

    Parameters
    ----------
    npoints : int
        number of points in the interpolation
    logf : array-like
        log of the frequencies
    cond : float, optional
        regularization condition of A matrix, by default 1e-5

    Returns
    -------
    Ai : array-like
        Inverse of A matrix
    b_0_coeff : float
        Coefficient for first bin
    b_last_coeff : float
        Coefficient for last bin
    """
    A = np.identity(npoints)
    np.fill_diagonal(A[:,1:], -(logf[1:-2]-logf[0:-3]) / (logf[2:-1] - logf[0:-3]))
    np.fill_diagonal(A[1:,:], -(logf[3:]-logf[2:-1]) / (logf[3:] - logf[1:-2]))
    U, s, Vt = sl.svd(A)
    reg = max(s) * cond
    Ai = np.dot(Vt.T / (s + reg), U.T)
    Ai_norm = np.array([Ai[:, ii] / np.max(Ai[:, ii]) for ii in range(Ai.shape[1])]).T
    maxvals = np.array([np.max(Ai[:, ii]) for ii in range(Ai.shape[1])]).squeeze()
    b_0_coeff = ((logf[2]-logf[1]) / (logf[2] - logf[0])) * maxvals[0]
    b_last_coeff = ((logf[-2]-logf[-3]) / (logf[-1] - logf[-3])) * maxvals[-1]
    return jnp.array(Ai_norm), jnp.array(b_0_coeff), jnp.array(b_last_coeff)

def create_amatrix_no_norm(npoints, logf, cond=0):
    """Create inverse of A matrix for inverting linear interpolation.
    Also return extra pieces needed to make interpolation correct at the edges.

    Parameters
    ----------
    npoints : int
        number of points in the interpolation
    logf : array-like
        log of the frequencies
    cond : float, optional
        regularization condition of A matrix, by default 1e-5

    Returns
    -------
    Ai : array-like
        Inverse of A matrix
    b_0_coeff : float
        Coefficient for first bin
    b_last_coeff : float
        Coefficient for last bin
    """
    A = np.identity(npoints)
    np.fill_diagonal(A[:,1:], -(logf[1:-2]-logf[0:-3]) / (logf[2:-1] - logf[0:-3]))
    np.fill_diagonal(A[1:,:], -(logf[3:]-logf[2:-1]) / (logf[3:] - logf[1:-2]))
    U, s, Vt = sl.svd(A)
    reg = max(s) * cond
    Ai = np.dot(Vt.T / (s + reg), U.T)
    b_0_coeff = ((logf[2]-logf[1]) / (logf[2] - logf[0]))
    b_last_coeff = ((logf[-2]-logf[-3]) / (logf[-1] - logf[-3]))
    return jnp.array(Ai), jnp.array(b_0_coeff), jnp.array(b_last_coeff)

def create_single_psr_freespec_model(psr, n_rn_frequencies, tnequad=False):
    """Create a numpyro model to sample free spectral red noise for a single pulsar

    Parameters
    ----------
    psr : ds.Pulsar
        discovery pulsar object
    n_rn_frequencies : int
        number of frequencies for red noise spectrum

    Returns
    -------
    numpyro model
        numpyro model for sampling
    psrl : ds.PulsarLikelihood
        pulsar likelihood object for this pulsar
    """
    Tspan = ds.getspan([psr])
    # create PTA model
    psrl = ds.PulsarLikelihood([psr.residuals,
                            ds.makenoise_measurement(psr, psr.noisedict, tnequad=tnequad),
                            ds.makegp_ecorr(psr, psr.noisedict),
                            ds.makegp_timing(psr),
                            ds.makegp_fourier(psr, ds.freespectrum, n_rn_frequencies, T=Tspan, name='red_noise')])

    # number of points for interpolation
    def uniform_model():
        f"""numpyro model for sampling free spectrum red noise for {psr.name}. It uses {n_rn_frequencies} frequencies for red noise spectrum.
        """
        log10_rho = numpyro.sample('log10_rho', dist.Uniform(-15, -4).expand([n_rn_frequencies]))
        numpyro.factor('ll', psrl.logL({f'{psr.name}_red_noise_log10_rho({n_rn_frequencies})': log10_rho}))
    return uniform_model, psrl

def create_single_psr_powerlaw_model(psr, n_rn_frequencies, tnequad=False):
    """Create numpyro model for sampling power law red noise for a single pulsar

    Parameters
    ----------
    psr : ds.Pulsar
        discovery pulsar object
    n_rn_frequencies : int
        number of frequencies for red noise spectrum

    Returns
    -------
    numpyro model
        numpyro model for sampling
    psrl : ds.PulsarLikelihood
        pulsar likelihood object for this pulsar
    """

    Tspan = ds.getspan([psr])
    # create PTA model
    psrl = ds.PulsarLikelihood([psr.residuals,
                            ds.makenoise_measurement(psr, psr.noisedict, tnequad=tnequad),
                            ds.makegp_ecorr(psr, psr.noisedict),
                            ds.makegp_timing(psr),
                            ds.makegp_fourier(psr, ds.powerlaw, n_rn_frequencies, T=Tspan, name='red_noise')])
    def powerlaw_model():
        f"""numpyro model for sampling power law red noise for {psr.name}. It uses {n_rn_frequencies} frequencies for red noise spectrum.
        """
        log10_A = numpyro.sample("log10_A", dist.Uniform(-20, -11))
        gamma = numpyro.sample("log10_gamma", dist.Uniform(0, 7))
        numpyro.factor('ll', psrl.logL({f'{psr.name}_red_noise_log10_A': log10_A, f'{psr.name}_red_noise_gamma': gamma}))
    return powerlaw_model, psrl

def create_rn_single_psr_model(psr, n_rn_frequencies, outliers=False, cond=1e-5, tnequad=False, tm_variance=None):
    """Create red noise flexible model for single pulsar

    Parameters
    ----------
    psr : ds.Pulsar
        pulsar object
    n_rn_frequencies : int
        number of frequencies for red noise spectrum
    outliers : bool, optional
        whether to include outliers in model [Not ], by default False
    cond : float, optional
        conditioning of linear interpolation matrix inverse, by default 1e-5

    Returns
    -------
    numpyro model
        numpyro model for sampling
    psrl : ds.PulsarLikelihood
        pulsar likelihood object for this pulsar
    """
    Tspan = ds.getspan([psr])
    # create PTA model
    psrl = ds.PulsarLikelihood([psr.residuals,
                            ds.makenoise_measurement(psr, psr.noisedict, tnequad=tnequad),
                            ds.makegp_ecorr(psr, psr.noisedict),
                            ds.makegp_timing(psr, variance=tm_variance),
                            ds.makegp_fourier(psr, ds.freespectrum, n_rn_frequencies, T=Tspan, name='red_noise')])

    # number of points for interpolation
    npoints_interp = n_rn_frequencies - 2
    freqs = jnp.arange(1, n_rn_frequencies + 1) / Tspan
    logf = jnp.log(freqs)

    # Precompute A and its inverse
    Ai, b_0_coeff, b_last_coeff = create_amatrix(npoints_interp, logf, cond=cond)


    def model(prior_dict=PRIOR_DICT, rng_key=None):
        f"""numpyro model for {psr.name}. It uses lncass for both the 
        linear interpolation inversion model and for deviations from that ("outliers").
        
        It uses {n_rn_frequencies} frequencies for red noise spectrum.

        Parameters
        ----------
        prior_dict : dictionary, optional
            prior dictionary for lncass parameters, by default PRIOR_DICT
        """

        beta = sample_lncass_single_psr(npoints_interp, tag='_rn', prior_dict=prior_dict, rng_key=rng_key)

        # sample first and last bins
        log_10_rho_0 = numpyro.sample("log_10_rho_0", dist.Uniform(-15, -4), rng_key=rng_key)
        log_10_rho_last = numpyro.sample("log_10_rho_last", dist.Uniform(-15, -4), rng_key=rng_key)

        # create b vector from precomputed coefficients
        b_0 = b_0_coeff * log_10_rho_0
        b_last = b_last_coeff * log_10_rho_last
        b = jnp.vstack([b_0, jnp.zeros((npoints_interp-2, 1)), b_last])

        # beta_o = sample_lncass_single_psr(npoints_interp, tag="_o", prior_dict=prior_dict,
        #                                   rng_key=rng_key)
        log10_rho_prime = jnp.dot(Ai, (jnp.atleast_2d(beta).T + b)) # + jnp.atleast_2d(beta_o).T
        log10_rho = numpyro.deterministic("log10_rho",
                                          jnp.clip(jnp.vstack([jnp.atleast_1d(log_10_rho_0), log10_rho_prime, jnp.atleast_1d(log_10_rho_last)]),
                                                  -15, -4))
        # jax.debug.breakpoint()
        numpyro.factor("ll", psrl.logL({f'{psr.name}_red_noise_log10_rho({n_rn_frequencies})': log10_rho}))

    return model, psrl


def create_rn_single_psr_model_sample_eigvecs(psr, n_rn_frequencies, outliers=False, cond=1e-5, tnequad=False):
    """Create red noise flexible model for single pulsar

    Parameters
    ----------
    psr : ds.Pulsar
        pulsar object
    n_rn_frequencies : int
        number of frequencies for red noise spectrum
    outliers : bool, optional
        whether to include outliers in model [Not ], by default False
    cond : float, optional
        conditioning of linear interpolation matrix inverse, by default 1e-5

    Returns
    -------
    numpyro model
        numpyro model for sampling
    psrl : ds.PulsarLikelihood
        pulsar likelihood object for this pulsar
    """
    Tspan = ds.getspan([psr])
    # create PTA model
    psrl = ds.PulsarLikelihood([psr.residuals,
                            ds.makenoise_measurement(psr, psr.noisedict, tnequad=tnequad),
                            ds.makegp_ecorr(psr, psr.noisedict),
                            ds.makegp_timing(psr),
                            ds.makegp_fourier(psr, ds.freespectrum, n_rn_frequencies, T=Tspan, name='red_noise')])

    # number of points for interpolation
    npoints_interp = n_rn_frequencies - 2
    freqs = jnp.arange(1, n_rn_frequencies + 1) / Tspan
    logf = jnp.log(freqs)

    # Precompute A and its inverse
    Ai, b_0_coeff, b_last_coeff = create_amatrix(npoints_interp, logf, cond=cond)
    out = jnp.linalg.eigh(Ai)
    Q = out.eigenvectors
    Sigma_inv = jnp.diag(out.eigenvalues**-1)

    beta_to_betaprime = jnp.dot(Q, Sigma_inv)

    def model(prior_dict=PRIOR_DICT, rng_key=None):
        f"""numpyro model for {psr.name}. It uses lncass for both the 
        linear interpolation inversion model and for deviations from that ("outliers").
        
        It uses {n_rn_frequencies} frequencies for red noise spectrum.

        Parameters
        ----------
        prior_dict : dictionary, optional
            prior dictionary for lncass parameters, by default PRIOR_DICT
        """

        beta = sample_lncass_single_psr(npoints_interp, tag='_rn', prior_dict=prior_dict, rng_key=rng_key)

        # sample first and last bins
        log_10_rho_0 = numpyro.sample("log_10_rho_0", dist.Uniform(-15, -4), rng_key=rng_key)
        log_10_rho_last = numpyro.sample("log_10_rho_last", dist.Uniform(-15, -4), rng_key=rng_key)

        # create b vector from precomputed coefficients
        b_0 = b_0_coeff * log_10_rho_0
        b_last = b_last_coeff * log_10_rho_last
        b = jnp.vstack([b_0, jnp.zeros((npoints_interp-2, 1)), b_last])

        # beta_o = sample_lncass_single_psr(npoints_interp, tag="_o", prior_dict=prior_dict,
        #                                   rng_key=rng_key)
        log10_rho_prime = jnp.dot(Q, (jnp.atleast_2d(beta).T + b)) # + jnp.atleast_2d(beta_o).T
        log10_rho = numpyro.deterministic("log10_rho",
                                          jnp.clip(jnp.vstack([jnp.atleast_1d(log_10_rho_0), log10_rho_prime, jnp.atleast_1d(log_10_rho_last)]),
                                                  -15, -4))
        
        numpyro.deterministic("beta_prime", jnp.dot(beta_to_betaprime, beta))
        # jax.debug.breakpoint()
        numpyro.factor("ll", psrl.logL({f'{psr.name}_red_noise_log10_rho({n_rn_frequencies})': log10_rho}))

    return model, psrl

def create_lncass_dm_pl_rn_model(psr, n_rn_frequencies=30, n_dm_frequencies=30, tnequad=False, cond=0):
    Tspan = ds.getspan([psr])
    psrl_nodmx = ds.PulsarLikelihood([psr.residuals,
                            ds.makenoise_measurement(psr, psr.noisedict, tnequad=tnequad),
                            ds.makegp_ecorr(psr, psr.noisedict),
                            ds.makegp_timing(psr),
                            ds.makegp_fourier(psr, ds.powerlaw, n_rn_frequencies, T=Tspan, name='red_noise'),
                            ds.makegp_fourier(psr, ds.freespectrum, n_dm_frequencies, T=Tspan, name='dm', fourierbasis=ds.dmfourierbasis)])
    # number of points for interpolation
    npoints_interp = n_dm_frequencies - 2
    freqs = jnp.arange(1, n_dm_frequencies + 1) / Tspan
    logf = jnp.log(freqs)

    # Precompute A and its inverse
    Ai, b_0_coeff, b_last_coeff = create_amatrix(npoints_interp, logf, cond=cond)
    def model(prior_dict=PRIOR_DICT, rng_key=None):
        
        beta = sample_lncass_single_psr(npoints_interp, tag='_dm', prior_dict=prior_dict, rng_key=rng_key)

        # sample first and last bins
        log_10_rho_0 = numpyro.sample("log_10_rho_0", dist.Uniform(-15, -4), rng_key=rng_key)
        log_10_rho_last = numpyro.sample("log_10_rho_last", dist.Uniform(-15, -4), rng_key=rng_key)

        # create b vector from precomputed coefficients
        b_0 = b_0_coeff * log_10_rho_0
        b_last = b_last_coeff * log_10_rho_last
        b = jnp.vstack([b_0, jnp.zeros((npoints_interp-2, 1)), b_last])

        beta_o = sample_lncass_single_psr(npoints_interp, tag="_o", prior_dict=prior_dict,
                                          rng_key=rng_key)
        log10_rho_prime = jnp.dot(Ai, (jnp.atleast_2d(beta).T + b)) + jnp.atleast_2d(beta_o).T
        log10_rho = numpyro.deterministic("log10_rho",
                                          jnp.clip(jnp.vstack([jnp.atleast_1d(log_10_rho_0), log10_rho_prime, jnp.atleast_1d(log_10_rho_last)]),
                                                  -15, -3))
        
        log10_A_rn = numpyro.sample("log10_A_rn", dist.Uniform(-20, -11), rng_key=rng_key)
        gamma_rn = numpyro.sample("gamma_rn", dist.Uniform(0, 7), rng_key=rng_key)
        # n_earth = numpyro.sample("n_earth", dist.Uniform(0, 10), rng_key=rng_key)
        # jax.debug.breakpoint()
        numpyro.factor("ll", psrl_nodmx.logL({f'{psr.name}_dm_log10_rho({n_dm_frequencies})': log10_rho,
                                              f'{psr.name}_red_noise_log10_A': log10_A_rn,
                                              f'{psr.name}_red_noise_gamma': gamma_rn}))

    return model, psrl_nodmx


def create_pl_dm_pl_rn_model(psr, n_rn_frequencies=30, n_dm_frequencies=30, tnequad=False, cond=0):
    Tspan = ds.getspan([psr])
    psrl_nodmx = ds.PulsarLikelihood([psr.residuals,
                            ds.makenoise_measurement(psr, psr.noisedict, tnequad=tnequad),
                            ds.makegp_ecorr(psr, psr.noisedict),
                            ds.makegp_timing(psr),
                            ds.makegp_fourier(psr, ds.powerlaw, n_rn_frequencies, T=Tspan, name='red_noise'),
                            ds.makegp_fourier(psr, ds.powerlaw, n_dm_frequencies, T=Tspan, name='dm', fourierbasis=ds.dmfourierbasis)])
    # number of points for interpolation
    freqs = jnp.arange(1, n_dm_frequencies + 1) / Tspan

    # Precompute A and its inverse
    def model(prior_dict=PRIOR_DICT, rng_key=None):

        # sample first and last bins
        log10_A_rn = numpyro.sample("log10_A_rn", dist.Uniform(-20, -11), rng_key=rng_key)
        gamma_rn = numpyro.sample("gamma_rn", dist.Uniform(0, 7), rng_key=rng_key)
        log10_A_dm = numpyro.sample("log10_A_dm", dist.Uniform(-20, -11), rng_key=rng_key)
        gamma_dm = numpyro.sample("gamma_dm", dist.Uniform(0, 7), rng_key=rng_key)
        numpyro.factor("ll", psrl_nodmx.logL({f'{psr.name}_red_noise_log10_A': log10_A_rn,
                                              f'{psr.name}_red_noise_gamma': gamma_rn,
                                              f'{psr.name}_dm_log10_A': log10_A_dm,
                                              f'{psr.name}_dm_gamma': gamma_dm,}))

    return model, psrl_nodmx

    

def create_rn_single_psr_model_javier(psr, n_rn_frequencies, cond=0):
    """Create red noise flexible model for single pulsar

    Parameters
    ----------
    psr : ds.Pulsar
        pulsar object
    n_rn_frequencies : int
        number of frequencies for red noise spectrum
    outliers : bool, optional
        whether to include outliers in model [Not ], by default False
    cond : float, optional
        conditioning of linear interpolation matrix inverse, by default 1e-5

    Returns
    -------
    numpyro model
        numpyro model for sampling
    psrl : ds.PulsarLikelihood
        pulsar likelihood object for this pulsar
    """
    Tspan = ds.getspan([psr])
    # create PTA model
    psrl = ds.PulsarLikelihood([psr.residuals,
                            ds.makenoise_measurement(psr, psr.noisedict),
                            ds.makegp_ecorr(psr, psr.noisedict),
                            ds.makegp_timing(psr),
                            ds.makegp_fourier(psr, ds.freespectrum, n_rn_frequencies, T=Tspan, name='red_noise')])

    # number of points for interpolation
    npoints_interp = n_rn_frequencies - 2
    freqs = jnp.arange(1, n_rn_frequencies + 1) / Tspan
    logf = jnp.log(freqs)

    # Precompute A and its inverse
    Ai, b_0_coeff, b_last_coeff = create_amatrix(npoints_interp, logf, cond=cond)


    def model(prior_dict=PRIOR_DICT, rng_key=None):
        f"""numpyro model for {psr.name}. It uses lncass for both the 
        linear interpolation inversion model and for deviations from that ("outliers").
        
        It uses {n_rn_frequencies} frequencies for red noise spectrum.

        Parameters
        ----------
        prior_dict : dictionary, optional
            prior dictionary for lncass parameters, by default PRIOR_DICT
        """

        beta = sample_lncass_single_psr_javier_method(npoints_interp, tag='_rn', prior_dict=prior_dict, rng_key=rng_key)

        # sample first and last bins
        log_10_rho_0 = numpyro.sample("log_10_rho_0", dist.Uniform(-15, -4), rng_key=rng_key)
        log_10_rho_last = numpyro.sample("log_10_rho_last", dist.Uniform(-15, -4), rng_key=rng_key)

        # create b vector from precomputed coefficients
        b_0 = b_0_coeff * log_10_rho_0
        b_last = b_last_coeff * log_10_rho_last
        b = jnp.vstack([b_0, jnp.zeros((npoints_interp-2, 1)), b_last])

        # beta_o = sample_lncass_single_psr(npoints_interp, tag="_o", prior_dict=prior_dict,
        #                                   rng_key=rng_key)
        log10_rho_prime = jnp.dot(Ai, (jnp.atleast_2d(beta).T + b)) # + jnp.atleast_2d(beta_o).T
        log10_rho = numpyro.deterministic("log10_rho",
                                          jnp.clip(jnp.vstack([jnp.atleast_1d(log_10_rho_0), log10_rho_prime, jnp.atleast_1d(log_10_rho_last)]),
                                                  -15, -4))
        # jax.debug.breakpoint()
        numpyro.factor("ll", psrl.logL({f'{psr.name}_red_noise_log10_rho({n_rn_frequencies})': log10_rho}))

    return model, psrl


def create_rn_single_psr_spline_model(psr, n_rn_frequencies, n_knots):
    """spline model for single pulsar

    Parameters
    ----------
    psr : ds.Pulsar
        pulsar object
    n_rn_frequencies : int
        number of red noise frequencies
    n_knots : int
        number of spline knots for interpolation

    Returns
    -------
    numpyro model   
        numpyro model for sampling
    psrl : ds.PulsarLikelihood
        pulsar likelihood object
    """
    Tspan = ds.getspan([psr])
    freqs = jnp.arange(1, n_rn_frequencies + 1) / Tspan
    fyr = 1 / 86400 / 365.25
    logf = jnp.log(freqs)
    # create PTA model
    psrl = ds.PulsarLikelihood([psr.residuals,
                            ds.makenoise_measurement(psr, psr.noisedict),
                            ds.makegp_ecorr(psr, psr.noisedict),
                            ds.makegp_timing(psr),
                            ds.makegp_fourier(psr, ds.freespectrum, n_rn_frequencies, T=Tspan, name='red_noise')])

    # number of points for interpolation

    def model(prior_dict=PRIOR_DICT, rng_key=None):
        f"""numpyro model that uses splines to estimate deviations from power law for {psr.name}.
        It has {n_knots} knots for interpolation. and {n_rn_frequencies} frequencies for red noise spectrum.

        Parameters
        ----------
        prior_dict : dictionary, optional
            prior dictionary for lncass parameters, by default PRIOR_DICT
        """

        beta = sample_lncass_single_psr(n_knots, tag='_rn', prior_dict=prior_dict, rng_key=rng_key)

        # initial power law        
        log10_A = numpyro.sample("log10_A", dist.Uniform(-20, -11), rng_key=rng_key)
        gamma = numpyro.sample("log10_gamma", dist.Uniform(0, 7), rng_key=rng_key)
        # rhos = numpyro.sample("log10_mean", dist.Uniform(-15, -4))
        rhos = jnp.log10(10**(2*log10_A) / (12 * np.pi**2) * (freqs / fyr)**(-gamma) * fyr**-3 / Tspan) / 2
        
        # deviations from power law
        # knots = jnp.linspace(logf[0], logf[-1], n_knots)
        knots = jnp.log(jnp.linspace(freqs[0], freqs[-1], n_knots))
        outliers = interp1d(logf, knots, beta, method='akima')
        log10_rho = numpyro.deterministic("log10_rho",
                                          rhos + outliers)
        # jax.debug.breakpoint()
        numpyro.deterministic("knots", knots)
        # numpyro.deterministic("pl_rhos", rhos)
        numpyro.factor("ll", psrl.logL({f'{psr.name}_red_noise_log10_rho({n_rn_frequencies})': log10_rho}))

    return model, psrl

def create_rn_pta_model(psrs, n_rn_frequencies, cond=1e-5,
                        array=False, tnequad=False,
                        simulate=False):
    """Create numpyro model for sampling and likelihood for full PTA. The intrinsic red noise and the GW
    are both modeled with the inverted interpolation model (using lncass for interpolation) combined with
    outliers (with lncass used to determine whether outliers are neded).

    Parameters
    ----------
    psrs : list
        List of ds.Pulsar objects for PTA
    n_rn_frequencies : int
        Number of red noise frequencies for the spectrum.
    cond : float, optional
        conditioning of linear inversion matrix, by default 1e-5

    Returns
    -------
    numpyro model
        numpyro model for sampling
    gl : ds.GlobalLikelihood
        global likelihood object for PTA
    """
    if simulate==True:
        variance=1e-20
    else:
        variance=None

    Tspan = ds.getspan(psrs)
    npsrs = len(psrs)
    # create PTA model
    if array:
        gl = ds.ArrayLikelihood((ds.PulsarLikelihood([psr.residuals,
                                                    ds.makenoise_measurement(psr, psr.noisedict, tnequad=tnequad),
                                                    ds.makegp_ecorr(psr, psr.noisedict),
                                                    ds.makegp_timing(psr, svd=True)]) for psr in psrs),
                              ds.makecommongp_fourier(psrs, ds.makefreespectrum_crn(n_rn_frequencies),
                                                 components={'log10_rho': n_rn_frequencies, 'crn_log10_rho': n_rn_frequencies},
                                                 T=Tspan, name='red_noise', common=['crn_log10_rho']))
    else:
        gl = ds.GlobalLikelihood([ds.PulsarLikelihood([psrs[ii].residuals,
                            ds.makenoise_measurement(psrs[ii], psrs[ii].noisedict,
                                                     tnequad=tnequad),
                            ds.makegp_ecorr(psrs[ii], psrs[ii].noisedict),
                            ds.makegp_timing(psrs[ii], variance=variance),
                            ds.makegp_fourier(psrs[ii], ds.freespectrum,
                                              n_rn_frequencies, T=Tspan,
                                              name='red_noise'),
                            ds.makegp_fourier(psrs[ii], ds.freespectrum,
                                              n_rn_frequencies, T=Tspan,
                                              name='gw', common=['gw_log10_rho'])
                            ]) for ii in range(len(psrs))])

    # number of points for interpolation
    npoints_interp = n_rn_frequencies - 2
    freqs = jnp.arange(1, n_rn_frequencies + 1) / Tspan
    logf = jnp.log(freqs)

    # Precompute A and its inverse
    Ai, b_0_coeff, b_last_coeff = create_amatrix(npoints_interp, logf, cond=cond)
    logL = transformations.simple_dict_transformation(gl.logL)


    def model(prior_dict=PRIOR_DICT, rng_key=None):
        f"""numpyro model for PTA with lncass for intrinsic red noise and GW for
        both the inverted linear interpolation model and for deviations from that ("outliers").
        
        There are {n_rn_frequencies} frequencies for red noise spectrum and {len(psrs)} pulsars in the PTA.

        Parameters
        ----------
        prior_dict : dictionary, optional
            prior dictionary for lncass parameters, by default PRIOR_DICT
        """

        # one extra for the gws
        beta = sample_lncass_pta(npoints_interp, npsrs+1, tag='_rn', prior_dict=prior_dict, rng_key=rng_key)

        # sample first and last bins
        log_10_rho_0 = numpyro.sample("log_10_rho_0", dist.Uniform(-15, -4).expand([npsrs+1]), rng_key=rng_key)
        log_10_rho_last = numpyro.sample("log_10_rho_last", dist.Uniform(-15, -4).expand([npsrs+1]), rng_key=rng_key)

        # create b vector from precomputed coefficients
        b_0 = b_0_coeff * log_10_rho_0
        b_last = b_last_coeff * log_10_rho_last
        b = jnp.vstack([b_0[None, :], jnp.zeros((npoints_interp-2, npsrs+1)), b_last[None, :]])
        # beta_o = sample_lncass_pta(npoints_interp, npsrs+1, tag="_o", prior_dict=prior_dict,
                                          # rng_key=rng_key)
        log10_rho_prime = jnp.dot(Ai, (jnp.atleast_2d(beta) + b)) # + jnp.atleast_2d(beta_o)
        log10_rho = numpyro.deterministic("log10_rho",
                                          jnp.clip(jnp.vstack([jnp.atleast_1d(log_10_rho_0), log10_rho_prime, jnp.atleast_1d(log_10_rho_last)]),
                                                  -15, -4))
        # resdict = {f'{psr.name}_red_noise_log10_rho({n_rn_frequencies})': log10_rho[:, ii] 
        numpyro.factor("ll", logL(log10_rho.T))

    return model, gl


def create_rn_pta_model_hierarchical(psrs, psr_freqs, psr_absspindowns, n_rn_frequencies, cond=1e-5, array=False, tnequad=False):
    """Create numpyro model for sampling and likelihood for full PTA. The intrinsic red noise and the GW
    are both modeled with the inverted interpolation model (using lncass for interpolation) combined with
    outliers (with lncass used to determine whether outliers are neded).

    Parameters
    ----------
    psrs : list
        List of ds.Pulsar objects for PTA
    n_rn_frequencies : int
        Number of red noise frequencies for the spectrum.
    cond : float, optional
        conditioning of linear inversion matrix, by default 1e-5

    Returns
    -------
    numpyro model
        numpyro model for sampling
    gl : ds.GlobalLikelihood
        global likelihood object for PTA
    """

    Tspan = ds.getspan(psrs)
    npsrs = len(psrs)
    # create PTA model
    if array:
        gl = ds.ArrayLikelihood((ds.PulsarLikelihood([psr.residuals,
                                                    ds.makenoise_measurement(psr, psr.noisedict, tnequad=tnequad),
                                                    ds.makegp_ecorr(psr, psr.noisedict),
                                                    ds.makegp_timing(psr, svd=True, constant=1e-6)]) for psr in psrs),
                              ds.makecommongp_fourier(psrs, ds.makefreespectrum_crn(n_rn_frequencies),
                                                 components={'log10_rho': n_rn_frequencies, 'crn_log10_rho': n_rn_frequencies},
                                                 T=Tspan, name='red_noise', common=['crn_log10_rho']))
    else:
        gl = ds.GlobalLikelihood([ds.PulsarLikelihood([psrs[ii].residuals,
                            ds.makenoise_measurement(psrs[ii], psrs[ii].noisedict, tnequad=tnequad),
                            ds.makegp_ecorr(psrs[ii], psrs[ii].noisedict),
                            ds.makegp_timing(psrs[ii]),
                            ds.makegp_fourier(psrs[ii], ds.freespectrum, n_rn_frequencies, T=Tspan, name='red_noise'),
                            ds.makegp_fourier(psrs[ii], ds.freespectrum, n_rn_frequencies, T=Tspan, name='gw', common=['gw_log10_rho'])
                            ]) for ii in range(len(psrs))])

    # number of points for interpolation
    npoints_interp = n_rn_frequencies - 2
    freqs = jnp.arange(1, n_rn_frequencies + 1) / Tspan
    logf = jnp.log(freqs)

    # Precompute A and its inverse
    Ai, b_0_coeff, b_last_coeff = create_amatrix(npoints_interp, logf, cond=cond)
    logL = transformations.simple_dict_transformation(gl.logL)


    def model(prior_dict=PRIOR_DICT, rng_key=None):
        # one extra for the gws
        beta = sample_lncass_pta(npoints_interp, npsrs+1, tag='_rn', prior_dict=prior_dict, rng_key=rng_key)

        # sample first and last bins
        log_10_rho_0 = numpyro.sample("log_10_rho_0", dist.Uniform(-15, -4).expand([npsrs+1]), rng_key=rng_key)
        log_10_rho_last = numpyro.sample("log_10_rho_last", dist.Uniform(-15, -4).expand([npsrs+1]), rng_key=rng_key)

        # create b vector from precomputed coefficients
        b_0 = b_0_coeff * log_10_rho_0
        b_last = b_last_coeff * log_10_rho_last
        b = jnp.vstack([b_0[None, :], jnp.zeros((npoints_interp-2, npsrs+1)), b_last[None, :]])
        beta_o = sample_lncass_pta(npoints_interp, npsrs+1, tag="_o", prior_dict=prior_dict,
                                          rng_key=rng_key)
        log10_rho_prime = jnp.dot(Ai, (jnp.atleast_2d(beta) + b)) + jnp.atleast_2d(beta_o)
        log10_rho = numpyro.deterministic("log10_rho",
                                          jnp.clip(jnp.vstack([jnp.atleast_1d(log_10_rho_0), log10_rho_prime, jnp.atleast_1d(log_10_rho_last)]), -30, -4))
        # Add hierarchical piece to our model. This is essentially a regression model on ln(sigma) in terms of ln(f), ln(\dot{f}).
        sigma_tn_psrs = numpyro.deterministic('sigma_tn_sq', jnp.array([trapezoid(10**(2 * log10_rho[:, ii]) * Tspan, x=freqs) for ii in range(npsrs)]))

        ln_sigma = jnp.log(jnp.sqrt(sigma_tn_psrs))
        
        log10_xi = numpyro.sample("log_xi", dist.Uniform(-10, 10))
        alpha = numpyro.sample("alpha", dist.Uniform(-5, 5))
        beta_exp = numpyro.sample("beta_exp", dist.Uniform(-5, 5))
        epsilon = numpyro.sample("epsilon", dist.Uniform(0, 5))

        chi = numpyro.deterministic("chi", 10**log10_xi * psr_freqs**alpha * psr_absspindowns**beta_exp)
        
        ln_prior = numpyro.deterministic("lnprior", jnp.sum(-0.5 * (ln_sigma - jnp.log(chi))**2 / (epsilon**2) - 0.5 * jnp.log(2 * jnp.pi * epsilon**2)))

        lnlike = numpyro.deterministic("lnlike", logL(log10_rho.T))
        
        numpyro.factor("lnpost", lnlike + ln_prior)

    return model, gl



def create_pta_model_plrn_fsgw(psrs, n_rn_frequencies, outliers=False, cond=1e-5, tnequad=False, array=True):
    Tspan = ds.getspan(psrs)
    freqs = jnp.arange(1, n_rn_frequencies + 1) / Tspan
    fyr = 1 / 86400 / 365.25
    npsrs = len(psrs)
    # create PTA model
    if array:
        gl = ds.ArrayLikelihood((ds.PulsarLikelihood([psr.residuals,
                                                    ds.makenoise_measurement(psr, psr.noisedict, tnequad=tnequad),
                                                    ds.makegp_ecorr(psr, psr.noisedict),
                                                    ds.makegp_timing(psr, svd=True, constant=1e-6)]) for psr in psrs),
                              ds.makecommongp_fourier(psrs, ds.makefreespectrum_crn(n_rn_frequencies),
                                                 components={'log10_rho': n_rn_frequencies, 'crn_log10_rho': n_rn_frequencies},
                                                 T=Tspan, name='red_noise', common=['crn_log10_rho']))
    else:
        gl = ds.GlobalLikelihood([ds.PulsarLikelihood([psrs[ii].residuals,
                            ds.makenoise_measurement(psrs[ii], psrs[ii].noisedict, tnequad=tnequad),
                            ds.makegp_ecorr(psrs[ii], psrs[ii].noisedict),
                            ds.makegp_timing(psrs[ii]),
                            ds.makegp_fourier(psrs[ii], ds.freespectrum, n_rn_frequencies, T=Tspan, name='red_noise'),
                            ds.makegp_fourier(psrs[ii], ds.freespectrum, n_rn_frequencies, T=Tspan, name='gw', common=['gw_log10_rho'])
                            ]) for ii in range(len(psrs))])
    logL = transformations.simple_dict_transformation(gl.logL)
    def model(prior_dict=PRIOR_DICT, rng_key=None):
        """numpyro model

        Parameters
        ----------
        prior_dict : dictionary, optional
            prior dictionary for lncass parameters, by default PRIOR_DICT
        """

        # one extra for the gws
        log10_rho_gw = numpyro.sample("log10_rho_gw", dist.Uniform(-15, -4).expand([n_rn_frequencies]), rng_key=rng_key)
        # rn A and Gamma
        log10_A_rn = numpyro.sample("log10_A_rn", dist.Uniform(-20, -11).expand([npsrs]), rng_key=rng_key)
        gamma_rn = numpyro.sample("log10_gamma_rn", dist.Uniform(0, 7).expand([npsrs]), rng_key=rng_key)
        resdict_log10_A = {f'{psr.name}_red_noise_log10_A': log10_A_rn[ii] for ii, psr in enumerate(psrs)}
        resdict_gamma = {f'{psr.name}_red_noise_gamma': gamma_rn[ii] for ii, psr in enumerate(psrs)}
        log10_rhos_rn = jnp.array([jnp.log10(10**(2*log10_A) / (12 * np.pi**2) * (freqs / fyr)**(-gamma) * fyr**-3 / Tspan) / 2
                for log10_A, gamma in zip(log10_A_rn, gamma_rn)])
        
        log10_rhos_total = jnp.vstack([log10_rhos_rn, log10_rho_gw])
        numpyro.factor("ll", logL(log10_rhos_total))
    return model, gl

def create_pta_model_plrn_plgw(psrs, n_rn_frequencies,
                               outliers=False, cond=0,
                               tnequad=False, simulate=True,
                               array=False):
    if simulate==True:
        variance=1e-20
    else:
        variance=None
    Tspan = ds.getspan(psrs)
    npsrs = len(psrs)
    # create PTA model
    if not array:
        gl = ds.GlobalLikelihood([ds.PulsarLikelihood([psrs[ii].residuals,
                                ds.makenoise_measurement(psrs[ii],
                                                        psrs[ii].noisedict,
                                                        tnequad=tnequad),
                                ds.makegp_ecorr(psrs[ii],
                                                psrs[ii].noisedict),
                                ds.makegp_timing(psrs[ii], variance=variance),
                                ds.makegp_fourier(psrs[ii], ds.powerlaw, n_rn_frequencies, T=Tspan, name='red_noise'),
                                ds.makegp_fourier(psrs[ii], ds.powerlaw, n_rn_frequencies, T=Tspan, name='gw', common=['gw_log10_A', 'gw_gamma'])
                                ]) for ii in range(len(psrs))])
    else:
        gl = ds.ArrayLikelihood((ds.PulsarLikelihood([psr.residuals,
                                                    ds.makenoise_measurement(psr, psr.noisedict, tnequad=tnequad),
                                                    ds.makegp_ecorr(psr, psr.noisedict),
                                                    ds.makegp_timing(psr, svd=True)]) for psr in psrs),
                                ds.makecommongp_fourier(psrs, ds.makepowerlaw_crn(n_rn_frequencies),
                                                        n_rn_frequencies, T=Tspan,
                                                        common=['crn_log10_A', 'crn_gamma'],
                                                        name='red_noise', vector=False))
    
    # makes things faster on GPU...not sure why
    logL = transformations.simple_dict_transformation(gl.logL)


    def model(prior_dict=PRIOR_DICT, rng_key=None):
        """numpyro model

        Parameters
        ----------
        prior_dict : dictionary, optional
            prior dictionary for lncass parameters, by default PRIOR_DICT
        """

        # one extra for the gws
        # log10_rho_gw = numpyro.sample("log10_rho_gw", dist.Uniform(-15, -4).expand([n_rn_frequencies]), rng_key=rng_key)
        # rn A and Gamma
        log10_A_rn = numpyro.sample("log10_A_rn", dist.Uniform(-20, -11).expand([npsrs+1]), rng_key=rng_key)
        gamma_rn = numpyro.sample("log10_gamma_rn", dist.Uniform(0, 7).expand([npsrs+1]), rng_key=rng_key)
        params = jnp.atleast_2d(jnp.array([[g, a] for g,a in zip(gamma_rn, log10_A_rn)]).flatten()).T
        numpyro.factor("ll", logL(params))
    return model, gl
