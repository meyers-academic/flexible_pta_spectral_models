
import jax
import jax.numpy as jnp
import numpy as np
import discovery as ds
import pandas as pd
import re
import os
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=4"
# from transformations import logit_transformations, transformed_posterior, transformed_posterior_dict, whitening_transformation, logit_transformation_dicts

def simple_dict_transformation(func):
    """change from dictionary as input to list of arrays as input

    Parameters
    ----------
    func : discovery likelihood
        discovery likelihood function
    """
    def to_dict(ys):
        xs = [y for y in ys.T]
        return dict(zip(func.params, jnp.array(xs).T))
    def transformed(ys):
        return func(to_dict(ys))
    return transformed