import matplotlib.pyplot as plt
plt.style.use("../../../data/meyers_latex.mplstyle")
import arviz as az
import numpy as np
import argparse
from pathlib import Path

psrname = 'J1909-3744'
run_name = 'rn_bpl_recovery'

results_dir = Path("../../../results") / run_name / psrname
plots_dir = Path("../../../plots") / run_name / psrname

plots_dir.mkdir(parents=True, exist_ok=True)

chain_file = Path(f"{results_dir}/{psrname}_rn_bpl_recovery.nc")

tmp = np.loadtxt(f"{results_dir}/{psrname}_rn_bpl_recovery_injection.txt")
freqs = tmp[:, 0]
injection = tmp[:, 1]

idata = az.from_netcdf(chain_file)

log10_rho = idata.posterior["log10_rho"].values.squeeze()
lambdas_rn = idata.posterior["lambdas_rn"].values.squeeze()
betas_rn = idata.posterior["beta_rn"].values.squeeze()
mu_lambda_rn = idata.posterior['mu_lambda_rn'].values.squeeze()
tau_rn = idata.posterior['tau_rn'].values.squeeze()

# violinplot for log10_rho
plt.figure(figsize=(6, 4))
parts = plt.violinplot(log10_rho, showextrema=False, positions=np.log10(freqs), widths=np.diff(np.log10(freqs))[15])
for b in parts['bodies']:
    b.set_alpha(0.8)
# plt.plot(np.log10(freqs),  , '-o', lw=1)
plt.plot(np.log10(freqs), injection, '-', lw=1, label='Injected', zorder=100)
plt.scatter(np.log10(freqs), injection, s=8, zorder=100, c='C1')
# plt.fill_between([], [], [], color='C0', alpha=0.8, label='Posterior')
plt.xlabel("$\log_{10}(f)$ [Hz]")
plt.ylabel("$\log_{10}(\\rho)$ [s]")
ax = plt.gca()
ax.tick_params(labelsize=22)
plt.ylim(-10, -4)
plt.legend()
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_single_pulsar_broken_powerlaw_injection_recovery.pdf")
plt.close()


plt.figure(figsize=(6, 4))
plt.hist(betas_rn[:, 4], bins=60, density=True)
plt.xlabel("$\\beta_5$", fontsize=26)
plt.ylabel('$p(\\beta_5 | \\delta t)$', fontsize=26)
ax = plt.gca()
ax.axvline(3, c='k', linestyle='--', lw=1.5)
ax.tick_params(labelsize=22)
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_single_pulsar_broken_powerlaw_beta_5_example.pdf")
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(betas_rn[:, 8], bins=60, density=True)
plt.xlabel("$\\beta_9$", fontsize=26)
plt.ylabel('$p(\\beta_9 | \\delta t)$', fontsize=26)
ax = plt.gca()
# ax.axvline(3, c='k', linestyle='--', lw=1.5)
ax.tick_params(labelsize=22)
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_single_pulsar_broken_powerlaw_beta_9_example.pdf")
plt.close()

plt.figure(figsize=(6, 4))
parts = plt.violinplot(lambdas_rn, showextrema=False, bw_method=0.001, widths=0.9)
for b in parts['bodies']:
    b.set_alpha(1)
plt.xlabel("$\lambda_i$", fontsize=26)
plt.ylabel("$p(\lambda_i | \delta t)$", fontsize=26)
ax = plt.gca()
ax.tick_params(labelsize=22)
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_single_pulsar_broken_powerlaw_lambda_posteriors.pdf")
plt.close()

plt.figure(figsize=(6, 4))
n, bins, patches = plt.hist(np.sum(lambdas_rn > 0.1, axis=1), bins=np.arange(0, 31)-0.5, edgecolor='w', density=True)
plt.xlabel("Number of $\\lambda_i > 0.1$")
ax = plt.gca()
ax.axvline(1, c='k', ls='--', lw=1.5)
ax.tick_params(labelsize=22)
plt.xticks(np.arange(0, 30, 5))
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_single_pulsar_broken_powerlaw_number_of_nodes.pdf")
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(mu_lambda_rn, bins=20, density=True, edgecolor='w')
plt.xlabel("$\mu_\lambda$", fontsize=26)
ax = plt.gca()
ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_single_pulsar_broken_powerlaw_mu_lambda.pdf")
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(tau_rn, bins=20, density=True, edgecolor='w')
plt.xlabel("$\\tau$", fontsize=26)
ax = plt.gca()
ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig(f"{plots_dir}/{psrname}_single_pulsar_broken_powerlaw_tau.pdf")
plt.close()