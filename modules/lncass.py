import numpyro.distributions as dist
import numpyro
from jax.scipy.special import expit, logsumexp, logit
import scipy.linalg as sl
import jax.numpy as jnp
import jax
import numpy as np
from interpax import interp2d

PRIOR_DICT = {'mu_lambda': dist.Uniform(-10, 10),
              'sigma_lambda': dist.Uniform(0, 10),
              'tau': dist.HalfCauchy(scale=1)}

def sample_lncass_single_psr(npoints, rng_key=None,
                             prior_dict=PRIOR_DICT, tag=''):

    tau = numpyro.sample(f"tau{tag}", prior_dict['tau'], rng_key=rng_key)
    mu_lambda = numpyro.sample(f"mu_lambda{tag}", prior_dict['mu_lambda'], rng_key=rng_key)
    sigma_lambda = numpyro.sample(f"sigma_lambda{tag}", prior_dict['sigma_lambda'], rng_key=rng_key)

    lambda_tilde = numpyro.sample(f"lambda_tilde{tag}", dist.Normal(mu_lambda, sigma_lambda).expand([npoints]), rng_key=rng_key)
    lambdas = numpyro.deterministic(f"lambdas{tag}", expit(lambda_tilde))

    beta_xi = numpyro.sample(f'beta_xi{tag}', dist.Normal(0, 1).expand([npoints]), rng_key=rng_key)
    beta = numpyro.deterministic(f'beta{tag}', beta_xi * tau * lambdas)
    return beta


def sample_lncass_single_psr_alt(npoints, rng_key=None,
                             prior_dict=PRIOR_DICT, tag=''):
    sigma_lambda = numpyro.sample(f"sigma_lambda{tag}", prior_dict['sigma_lambda'], rng_key=rng_key)
    tau = numpyro.sample(f"tau{tag}", prior_dict['tau'], rng_key=rng_key)
    # this is the parameter that seems best constrained at this point
    snr_lambda = numpyro.sample(f"snr_lambda{tag}", dist.Uniform(-10, 10), rng_key=rng_key)
    mu_lambda = numpyro.deterministic(f"mu_lambda{tag}", snr_lambda * sigma_lambda)
    # sigma_lambda = numpyro.sample(f"sigma_lambda{tag}", prior_dict['sigma_lambda'], rng_key=rng_key)

    lambda_tilde = numpyro.sample(f"lambda_tilde{tag}", dist.Normal(mu_lambda, sigma_lambda).expand([npoints]), rng_key=rng_key)
    lambdas = numpyro.deterministic(f"lambdas{tag}", expit(lambda_tilde))

    beta_xi = numpyro.sample(f'beta_xi{tag}', dist.Normal(0, 1).expand([npoints]), rng_key=rng_key)
    beta = numpyro.deterministic(f'beta{tag}', beta_xi * tau * lambdas)
    return beta

def setup_javier_sampling_method(spread):

    def u_to_beta(u, alpha):
        if alpha < 0.5:
            alpha = 1-alpha
        if u < alpha / 2:
            beta = -spread / (alpha / 2) * u + spread
        elif u > alpha / 2 and u < (1-alpha / 2):
            beta = 0
        elif u > (1 - alpha / 2):
            beta = -spread / (alpha / 2) * u + spread * (2/alpha - 1)
        return beta

    # jax version of u_to_beta using jax.lax.cond that doesn't use anonymous functions
    N_uvals =200
    N_alphas = 201
    beta_vals = np.zeros((N_uvals, N_alphas))
    alphas = np.zeros_like(beta_vals)
    uvals = np.zeros_like(beta_vals)
    for ii, u in enumerate(np.linspace(0, 1, N_uvals)):
        for jj, alpha in enumerate(np.linspace(0, 1, N_alphas)):
            beta = u_to_beta(u, alpha)
            beta_vals[ii, jj] = beta
            alphas[ii,jj] = alpha
            uvals[ii,jj] = u

    alphas = jnp.linspace(0, 1, N_alphas)
    uvals =  jnp.linspace(0, 1, N_uvals)
    heights = jnp.array(beta_vals)
    return alphas, uvals, heights

# TODO: fix this. sets prior from -5 to 5, should probably be tunable.
alphas, uvals, heights = setup_javier_sampling_method(5)


def sample_lncass_single_psr_javier_method(npoints, rng_key=None, tag='', prior_dict=PRIOR_DICT):
    # alpha = numpyro.sample(f'alpha{tag}', dist.Uniform(0, 1), rng_key=rng_key)
    alpha = 0.1
    u = numpyro.sample(f'u{tag}', dist.Uniform(0, 1).expand([npoints]), rng_key=rng_key)
    beta = numpyro.deterministic(f'beta{tag}',interp2d(alpha*jnp.ones(npoints), u, alphas, uvals, heights.T))
    return beta

def sample_lncass_pta(npoints, npsrs, rng_key=None,
                             prior_dict=PRIOR_DICT, tag=''):

    tau = numpyro.sample(f"tau{tag}", prior_dict['tau'].expand([npsrs]), rng_key=rng_key)
    mu_lambda = numpyro.sample(f"mu_lambda{tag}", prior_dict['mu_lambda'].expand([npsrs]), rng_key=rng_key)
    sigma_lambda = numpyro.sample(f"sigma_lambda{tag}", prior_dict['sigma_lambda'].expand([npsrs]), rng_key=rng_key)

    lambda_tilde = numpyro.sample(f"lambda_tilde{tag}", dist.Normal(mu_lambda, sigma_lambda).expand([npoints, npsrs]), rng_key=rng_key)
    lambdas = numpyro.deterministic(f"lambdas{tag}", expit(lambda_tilde))

    beta_xi = numpyro.sample(f'beta_xi{tag}', dist.Normal(0, 1).expand([npoints, npsrs]), rng_key=rng_key)
    beta = numpyro.deterministic(f'beta{tag}', beta_xi * tau * lambdas)
    return beta