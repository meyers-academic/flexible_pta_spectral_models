import os, re

import discovery as ds
# import PTMCMCSampler


import numpy as np

import jax
import jax.numpy as jnp
import glob
import matplotlib.pyplot as plt
import numpyro
from numpyro import distributions as dist, infer
import sys
sys.path.append("../modules")
import models as lncass_models
from lncass import PRIOR_DICT
import argparse
import json
import arviz
from loguru import logger
from pathlib import Path

argparser = argparse.ArgumentParser()
argparser.add_argument('--seed-injection', type=int, default=0)
argparser.add_argument('--seed-sampling', type=int, default=1)
argparser.add_argument("--pulsar-name", type=str, default="J1909-3744")
argparser.add_argument("--data-dir", type=str, default="../data")
argparser.add_argument("--noisedict", type=str, default="../data/channelized_12p5yr_v3_full_noisedict.json")
argparser.add_argument("--npoints-sampling", type=int, default=1000)
argparser.add_argument("--npoints-warmup", type=int, default=1000)
argparser.add_argument("--num-rn-frequencies", type=int, default=30)
argparser.add_argument("--max-tree-depth", type=int, default=10)
argparser.add_argument("--target-accept-prob", type=float, default=0.95)
argparser.add_argument("--results-directory", type=str, default="../results")
argparser.add_argument("--run-name", type=str, default="rn_bpl_recovery")

def load_data_and_noise(data_dir, noisedict, pulsar_name):
    with open(noisedict, "r") as f:
        noisedict = json.load(f)
    psr = ds.Pulsar.read_feather(os.path.join(data_dir, f"{pulsar_name}-12p5.feather"))
    psr.noisedict = noisedict
    return psr

def run(params):
    logger.info(f"Running broken powerlaw injection for {params.pulsar_name}")
    psr = load_data_and_noise(params.data_dir, params.noisedict, params.pulsar_name)
    # for simulation
    _, gl_sim = lncass_models.create_rn_single_psr_model(psr, params.num_rn_frequencies, cond=0, tnequad=True, tm_variance=1e-20)
    logger.info(f"Simulating data for {params.pulsar_name}")
    
    Tspan_psr = ds.getspan([psr])
    
    freqs = jnp.arange(1,  params.num_rn_frequencies + 1) / Tspan_psr
    logf = jnp.log(freqs)
    npoints_interp =  params.num_rn_frequencies - 2

        # Precompute A and its inverse
    b_0 = -6
    b_last = -9
    Ai, b_0_coeff, b_last_coeff = lncass_models.create_amatrix(npoints_interp, logf, cond=0)

    # create injection
    bvec = np.r_[b_0_coeff*b_0, np.zeros(npoints_interp - 2), b_last_coeff*b_last]
    bvec[4] = 3
    injection = np.r_[b_0, Ai @ bvec, b_last]
    
    _, residuals = gl_sim.sample(jax.random.PRNGKey(params.seed_injection), {f'{params.pulsar_name}_red_noise_log10_rho(30)': injection})
    
    
    psr.residuals = np.array(residuals)
    
    del gl_sim
    
    logger.success(f"Data simulated for {params.pulsar_name}")
    logger.info("Creating model for sampling")
    
    model, gl = lncass_models.create_rn_single_psr_model(psr, params.num_rn_frequencies, cond=0, tnequad=True)
    
    logger.success("Model created")
    logger.info(f"Sampling for {params.pulsar_name} with {params.npoints_warmup} warmup points and {params.npoints_sampling} sampling points")
    
    lncass_outliers_sampler = infer.MCMC(
    infer.NUTS(model, max_tree_depth=params.max_tree_depth, target_accept_prob=params.target_accept_prob),
    num_warmup=params.npoints_warmup,
    num_samples=params.npoints_sampling,
    num_chains=1,
    progress_bar=True,chain_method='vectorized')
    

    prior_dict = {**PRIOR_DICT, 'sigma_lambda':dist.Uniform(14.999, 15.001), 'mu_lambda': dist.Uniform(-50, 50)}

    lncass_outliers_sampler.run(jax.random.PRNGKey(0), prior_dict=prior_dict)
    
    lncass_outliers_sampler.print_summary()
    
    idata = arviz.from_numpyro(lncass_outliers_sampler)
    results_dir = Path(params.results_directory).joinpath(params.run_name).joinpath(params.pulsar_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    injection_file = results_dir.joinpath(f"{params.pulsar_name}_{params.run_name}_injection.txt")
    np.savetxt(injection_file, np.c_[freqs, injection])
    
    outchain_file = results_dir.joinpath(f"{params.pulsar_name}_{params.run_name}.nc")
    
    logger.info(f"Saving data to netcdf here: {outchain_file}")
    
    if outchain_file.exists():
        outchain_file.unlink()
        logger.warning(f"Overwriting existing file {outchain_file}")
    idata.to_netcdf(outchain_file, overwrite_existing=True)
    
    logger.success(f"Sampling results saved for {params.pulsar_name}.")
    
if __name__ == "__main__":
    args = argparser.parse_args()
    run(args)