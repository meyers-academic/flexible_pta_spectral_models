import numpyro.distributions as dist
import numpyro
from jax.scipy.special import expit, logsumexp, logit
import scipy.linalg as sl
import jax.numpy as jnp
import jax

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