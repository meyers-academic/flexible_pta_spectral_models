
import jax
import jax.numpy as jnp
import numpy as np
import discovery as ds
import pandas as pd
import re
import os
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=4"
# from transformations import logit_transformations, transformed_posterior, transformed_posterior_dict, whitening_transformation, logit_transformation_dicts


if jax.config.jax_enable_x64:
  print("X64 enabled")

  priordict_standard = {
      ".*_rednoise_log10_A": [-20, -11],
      ".*_rednoise_gamma": [0, 7],
      ".*_red_noise_log10_A": [-20, -11],  # deprecated
      ".*_red_noise_gamma": [0, 7],  # deprecated
      "crn_log10_A": [-18, -11],
      "crn_gamma": [0, 7],
      "gw_(.*_)?log10_A": [-18, -11],
      "gw_(.*_)?gamma": [0, 7],
      ".*_red_noise_log10_rho": [-9, -4],
      "dmgp_log10_A": [-20, -11],
      "dmgp_gamma": [0, 7],
      "crn_log10_rho": [-9, -4],
      "gw_(.*_)?log10_rho": [-9, -4],
  }
else:
  priordict_standard = {
      # ".*_rednoise_log10_A": [-20, -11],
      # ".*_rednoise_gamma": [0, 7],
      # ".*_red_noise_log10_A": [-20, -11],  # deprecated
      ".*_red_noise_log10_A": [-18, -12],  # deprecated
      ".*_red_noise_gamma": [0, 7],  # deprecated
      "crn_log10_A": [-18, -11],
      "crn_gamma": [0, 7],
      # "gw_(.*_)?log10_A": [-18, -11],
      "gw_(.*_)?log10_A": [-18, -12],
      "gw_(.*_)?gamma": [0, 7],
      "dmgp_log10_A": [-20, -11],
      "dmgp_gamma": [0, 7],
      "crn_log10_rho": [-9, -4],
      "gw_(.*_)?log10_rho": [-9, -4],
  }

matrix = ds.matrix

def makelogtransform_uniform(func, priordict={}):
    priordict = {**priordict_standard, **priordict}

    a, b = [], []
    for par in func.params:
        for pname, prange in priordict.items():
            if re.match(pname, par):
                a.append(prange[0])
                b.append(prange[1])
                break
        else:
            raise KeyError(f"No known prior for {par}.")

    a, b = matrix.jnparray(a), matrix.jnparray(b)

    def to_dict(ys):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))
        return dict(zip(func.params, xs))

    def to_vec(params):
        xs = matrix.jnparray([params[pname] for pname in func.params])
        return jnp.arctanh((a + b - 2*xs)/(a - b))

    def to_df(ys):
        xs = 0.5 * (b + a + (b - a) * jnp.tanh(ys))
        return pd.DataFrame(np.array(xs), columns=func.params)

    def prior(ys):
        return jnp.sum(jnp.log(2.0) - 2.0 * jnp.logaddexp(ys, -ys))

        # return jnp.sum(jnp.log(0.5) - 2.0 * jnp.log(jnp.cosh(ys)))
        # but   log(0.5) - 2 * log(cosh(y))
        #     = log(0.5) - 2 * log((exp(x) + exp(-x))/2)
        #     = log(0.5) - 2 * (log(exp(x) - exp(-x)) - log(2.0))
        #     = log(2.0) - 2 * logaddexp(x, -x)

    def transformed(ys):
        return func(to_dict(ys)) + prior(ys)

    transformed.params = func.params

    transformed.prior = prior
    transformed.to_dict = to_dict
    transformed.to_vec = to_vec
    transformed.to_df = to_df

    return transformed