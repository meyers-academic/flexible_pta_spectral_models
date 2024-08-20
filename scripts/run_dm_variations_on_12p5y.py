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
from dmx_utils import PulsarNoDMX
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
argparser.add_argument("--data-dir", type=str, default="/Users/patrickmeyers/Documents/Documents - Patrick’s MacBook Pro/Projects/pulsar_discovery_utilities/dmgp_tests/data")
argparser.add_argument("--noisedict", type=str, default=None)
argparser.add_argument("--npoints-sampling", type=int, default=1000)
argparser.add_argument("--npoints-warmup", type=int, default=1000)
argparser.add_argument("--num-rn-frequencies", type=int, default=30)
argparser.add_argument("--num-dm-frequencies", type=int, default=30)
argparser.add_argument("--max-tree-depth", type=int, default=10)
argparser.add_argument("--target-accept-prob", type=float, default=0.99)
argparser.add_argument("--results-directory", type=str, default="../results")
argparser.add_argument("--tnequad", default=False, action='store_true')
argparser.add_argument("--run-name", type=str, default="dm_recovery")

def load_data_and_noise(data_dir, noisedict, pulsar_name):

    psrs_nodmx = [PulsarNoDMX.read_feather(psrfile) for psrfile in sorted(glob.glob(f'{data_dir}/*[JB]*.feather'))]
    names = [psr.name for psr in psrs_nodmx]
    idx = names.index(pulsar_name)
    psr_nodmx = psrs_nodmx[idx]
    if noisedict is not None:
        with open(noisedict, "r") as f:
            noisedict = json.load(f)
        psr_nodmx.noisedict = noisedict
    return psr_nodmx

def run(params):
    logger.info(f"Analyzing DM Variations for {params.pulsar_name}")
    psr = load_data_and_noise(params.data_dir, params.noisedict, params.pulsar_name)
    Tspan = ds.getspan([psr])
    freqs = jnp.arange(1,  params.num_dm_frequencies + 1) / Tspan
    logger.info("Creating model for sampling")
    
    model, gl = lncass_models.create_lncass_dm_pl_rn_model(psr, n_rn_frequencies=params.num_rn_frequencies, cond=0, tnequad=params.tnequad, n_dm_frequencies=params.num_dm_frequencies)
    
    logger.success("Model created")
    logger.info(f"Sampling for {params.pulsar_name} with {params.npoints_warmup} warmup points and {params.npoints_sampling} sampling points")
    
    lncass_outliers_sampler = infer.MCMC(
    infer.NUTS(model, max_tree_depth=params.max_tree_depth, target_accept_prob=params.target_accept_prob),
    num_warmup=params.npoints_warmup,
    num_samples=params.npoints_sampling,
    num_chains=1,
    progress_bar=True,chain_method='vectorized')
    

    prior_dict = {**PRIOR_DICT, 'sigma_lambda':dist.Uniform(14.999, 15.001), 'mu_lambda': dist.Uniform(-100, 100)}

    lncass_outliers_sampler.run(jax.random.PRNGKey(0), prior_dict=prior_dict)
    
    lncass_outliers_sampler.print_summary()
    
    idata = arviz.from_numpyro(lncass_outliers_sampler)

    
    results_dir = Path(params.results_directory).joinpath(params.run_name).joinpath(params.pulsar_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    outchain_file = results_dir.joinpath(f"{params.pulsar_name}_{params.run_name}.nc")
    
    np.savetxt(results_dir.joinpath(f"{params.pulsar_name}_{params.run_name}_freqs.txt"), freqs)
    
    logger.info(f"Saving data to netcdf here: {outchain_file}")
    
    if outchain_file.exists():
        outchain_file.unlink()
        logger.warning(f"Overwriting existing file {outchain_file}")
    idata.to_netcdf(outchain_file, overwrite_existing=True)
    
    logger.success(f"Sampling results saved for {params.pulsar_name}.")
    
if __name__ == "__main__":
    args = argparser.parse_args()
    run(args)