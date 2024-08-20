import matplotlib.pyplot as plt
plt.style.use("../../../data/meyers_latex.mplstyle")
import arviz as az
import numpy as np
import argparse
from pathlib import Path

psrname = 'J0613-0200'
run_name = 'dm_recovery'

results_dir = Path("../../../results") / run_name / psrname
plots_dir = Path("../../../plots") / run_name / psrname
freqs = np.loadtxt(f"{results_dir}/{psrname}_{run_name}_freqs.txt")

plots_dir.mkdir(parents=True, exist_ok=True)

chain_file = Path(f"{results_dir}/{psrname}_{run_name}.nc")

idata = az.from_netcdf(chain_file)

log10_rho = idata.posterior["log10_rho"].values.squeeze()

lambdas_dm = idata.posterior["lambdas_dm"].values.squeeze()
betas_dm = idata.posterior["beta_dm"].values.squeeze()
mu_lambda_dm = idata.posterior['mu_lambda_dm'].values.squeeze()
tau_dm = idata.posterior['tau_dm'].values.squeeze()

lambdas_o = idata.posterior["lambdas_o"].values.squeeze()
betas_o = idata.posterior["beta_o"].values.squeeze()
mu_lambda_o = idata.posterior['mu_lambda_o'].values.squeeze()
tau_o = idata.posterior['tau_o'].values.squeeze()

# violinplot for log10_rho
plt.figure(figsize=(6, 4))
parts = plt.violinplot(log10_rho, showextrema=False, positions=np.log10(freqs), widths=np.diff(np.log10(freqs))[15])
for b in parts['bodies']:
    b.set_alpha(0.8)
plt.xlabel("$\log_{10}(f)$ [Hz]")
plt.ylabel("$\log_{10}(\\rho)$ [s]")
ax = plt.gca()
ax.tick_params(labelsize=22)
plt.ylim(-10, -4)
plt.legend()
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_single_pulsar_dm_recovery.pdf")
plt.close()

plt.figure(figsize=(6, 4))
parts = plt.violinplot(lambdas_dm, showextrema=False, bw_method=0.001, widths=0.9)
for b in parts['bodies']:
    b.set_alpha(1)
plt.xlabel("$\lambda_i$", fontsize=26)
plt.ylabel("$p(\lambda_i | \delta t)$", fontsize=26)
ax = plt.gca()
ax.tick_params(labelsize=22)
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_single_pulsar_dm_lambdas_posteriors.pdf")
plt.close()

plt.figure(figsize=(6, 4))
parts = plt.violinplot(lambdas_o, showextrema=False, bw_method=0.001, widths=0.9)
for b in parts['bodies']:
    b.set_alpha(1)
plt.xlabel("$\lambda_i$", fontsize=26)
plt.ylabel("$p(\lambda_i | \delta t)$", fontsize=26)
ax = plt.gca()
ax.tick_params(labelsize=22)
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_single_pulsar_dm_lambdas_outlier_posteriors.pdf")
plt.close()

plt.figure(figsize=(6, 4))
n, bins, patches = plt.hist(np.sum(lambdas_dm > 0.1, axis=1), bins=np.arange(0, 31)-0.5, edgecolor='w', density=True)
plt.xlabel("Number of $\\lambda_i > 0.1$")
ax = plt.gca()
ax.axvline(1, c='k', ls='--', lw=1.5)
ax.tick_params(labelsize=22)
plt.xticks(np.arange(0, 30, 5))
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_single_pulsar_dm_number_of_nodes.pdf")
plt.close()

plt.figure(figsize=(6, 4))
n, bins, patches = plt.hist(np.sum(lambdas_o > 0.1, axis=1), bins=np.arange(0, 31)-0.5, edgecolor='w', density=True)
plt.xlabel("Number of $\\lambda_{o,i} > 0.1$")
ax = plt.gca()
ax.axvline(1, c='k', ls='--', lw=1.5)
ax.tick_params(labelsize=22)
plt.xticks(np.arange(0, 30, 5))
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_single_pulsar_dm_outliers_number_of_nodes.pdf")
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(mu_lambda_dm, bins=20, density=True, edgecolor='w')
plt.xlabel("$\mu_{\lambda,dm}$", fontsize=26)
ax = plt.gca()
ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_single_pulsar_dm_mu_lambda.pdf")
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(mu_lambda_dm, bins=20, density=True, edgecolor='w')
plt.xlabel("$\mu_{\lambda,o}$", fontsize=26)
ax = plt.gca()
ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_single_pulsar_dm_mu_lambda_outlier.pdf")
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(tau_dm, bins=20, density=True, edgecolor='w')
plt.xlabel("$\\tau_{dm}$", fontsize=26)
ax = plt.gca()
ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_dm_tau.pdf")
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(tau_o, bins=20, density=True, edgecolor='w')
plt.xlabel("$\\tau_{o}$", fontsize=26)
ax = plt.gca()
ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_dm_outlier_tau.pdf")
plt.close()