# ============================================================
# 03 - Bayesian Model — EXERCISE GUIDE
# 
# Goal: Estimate seroconversion rates and treatment effect
#       using a Beta-Binomial model in PyMC
#       Answer: P(OR < 1 | data) — what frequentist p-values can't
# ============================================================

# ---- SETUP: imports ----

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az 
import pymc as pm

# =============================================================
# SECTION 1: Load the data and extract counts
# =============================================================

df = pd.read_csv("../data/simulated_patient_data.csv")

print(df.head(2))

# For each group (season x treatment), extract:
#   - n_total
#   - n_seroconverted
summary = df.groupby(['season','treatment']).agg(n_total = ('participant_id','count'),n_seroconverted = ('seroconverted','sum')).reset_index()

print(summary)


# Create 4 variables:
#   n_s2_ad, k_s2_ad       (n total, n successes for S2 A&D)
#   n_s2_pl, k_s2_pl       (S2 Placebo)
#   n_s3_ad, k_s3_ad       (S3 A&D)
#   n_s3_pl, k_s3_pl       (S3 Placebo)

n_s2_ad = summary[(summary['season']=='S2') & (summary['treatment'] == 'A&D')].iloc[0]['n_total']
k_s2_ad = summary[(summary['season']=='S2')&(summary['treatment']=='A&D')].iloc[0]['n_seroconverted']
n_s2_pl = summary[(summary['season']=='S2') & (summary['treatment'] == 'Placebo')].iloc[0]['n_total']
k_s2_pl = summary[(summary['season']=='S2')&(summary['treatment']=='Placebo')].iloc[0]['n_seroconverted']

n_s3_ad = summary[(summary['season']=='S3') & (summary['treatment'] == 'A&D')].iloc[0]['n_total']
k_s3_ad = summary[(summary['season']=='S3')&(summary['treatment']=='A&D')].iloc[0]['n_seroconverted']
n_s3_pl = summary[(summary['season']=='S3') & (summary['treatment'] == 'Placebo')].iloc[0]['n_total']
k_s3_pl = summary[(summary['season']=='S3')&(summary['treatment']=='Placebo')].iloc[0]['n_seroconverted']


# =============================================================
# SECTION 2: Build the Bayesian model (uninformative prior)
# =============================================================
with pm.Model() as model_uninformative:
    #1 define prior distribution from unkown parameters based on assumption beta(1,1)
    p_s2_ad = pm.Beta("p_s2_ad", alpha=1, beta=1)
    p_s2_pl = pm.Beta("p_s2_pl", alpha =1, beta=1)
    p_s3_ad = pm.Beta("p_s3_ad", alpha =1, beta=1)
    p_s3_pl = pm.Beta("p_s3_pl", alpha =1, beta=1)

    #2 define likelihood based on observed data 
    pm.Binomial('obs_s2_ad', n=n_s2_ad, p=p_s2_ad, observed = k_s2_ad)
    pm.Binomial('obs_s2_pl', n=n_s2_pl, p=p_s2_pl, observed = k_s2_pl)
    pm.Binomial('obs_s3_ad', n=n_s3_ad, p=p_s3_ad, observed = k_s3_ad)
    pm.Binomial('obs_s3_pl', n=n_s3_pl, p=p_s3_pl, observed = k_s3_pl)

    #3 compute ORs
    or_s2 = pm.Deterministic("or_s2", (p_s2_ad/(1-p_s2_ad)) / (p_s2_pl/(1-p_s2_pl)))
    or_s3 = pm.Deterministic("or_s3", (p_s3_ad/(1-p_s3_ad)) / (p_s3_pl/(1-p_s3_pl)))

    print(or_s2, or_s3)
    
    #4 sample
    trace_uninformative = pm.sample(2000, return_inferencedata=True)  


# =============================================================
# SECTION 3: Inspect the results
# =============================================================
#summary_table = az.summary(trace_uninformative, var_names=["p_s2_ad", "p_s2_pl", "p_s3_ad", "p_s3_pl", "or_s2", "or_s3"])
 
#print(summary_table)


# =============================================================
# SECTION 4: The key Bayesian question
# =============================================================
# For each season, compute:
#   P(OR < 1 | data) = probability A&D has LOWER odds than Placebo
p_or_s2 = (trace_uninformative.posterior['or_s2'].values < 1.0).mean()
p_or_s3 = (trace_uninformative.posterior['or_s3'].values < 1.0).mean()

#print(f"posterior mean OR of S2 is {p_or_s2} and of S3 is {p_or_s3}")

# Compute credible intervals 

print(az.hdi(trace_uninformative.posterior["or_s2"], hdi_prob=0.95)["or_s2"].values)
print(az.hdi(trace_uninformative.posterior["or_s3"], hdi_prob=0.95)["or_s3"].values)
      

# =============================================================
# SECTION 5: Repeat with informed prior
# =============================================================
# Run the same model but with Beta(8, 2) as the prior for all groups
# (belief that baseline seroconversion rate is ~80%)
#
with pm.Model() as model_informative:
    #1 define prior distribution from informed parameters based on assumption beta(8,2)
    p_s2_ad = pm.Beta("p_s2_ad", alpha=8, beta=2)
    p_s2_pl = pm.Beta("p_s2_pl", alpha =8, beta=2)
    p_s3_ad = pm.Beta("p_s3_ad", alpha =8, beta=2)
    p_s3_pl = pm.Beta("p_s3_pl", alpha =8, beta=2)

    #2 define likelihood based on observed data 
    pm.Binomial('obs_s2_ad', n=n_s2_ad, p=p_s2_ad, observed = k_s2_ad)
    pm.Binomial('obs_s2_pl', n=n_s2_pl, p=p_s2_pl, observed = k_s2_pl)
    pm.Binomial('obs_s3_ad', n=n_s3_ad, p=p_s3_ad, observed = k_s3_ad)
    pm.Binomial('obs_s3_pl', n=n_s3_pl, p=p_s3_pl, observed = k_s3_pl)

    #3 compute ORs
    or_s2 = pm.Deterministic("or_s2", (p_s2_ad/(1-p_s2_ad)) / (p_s2_pl/(1-p_s2_pl)))
    or_s3 = pm.Deterministic("or_s3", (p_s3_ad/(1-p_s3_ad)) / (p_s3_pl/(1-p_s3_pl)))
    
    #4 sample
    trace_informative = pm.sample(2000, return_inferencedata=True)  

p_or_s2_informed = (trace_informative.posterior['or_s2'].values < 1.0).mean()
p_or_s3_informed = (trace_informative.posterior['or_s3'].values < 1.0).mean()

#print(f"posterior mean OR of S2 is {p_or_s2} and of S3 is {p_or_s3}")

# Compute credible intervals 

print(az.hdi(trace_informative.posterior["or_s2"], hdi_prob=0.95)["or_s2"].values)
print(az.hdi(trace_informative.posterior["or_s3"], hdi_prob=0.95)["or_s3"].values)

print(f"Uninformative prior: P(OR < 1 | data) = {p_or_s2:.3f} for S2 and {p_or_s3:.3f} for S3")
print(f"Informed prior:      P(OR < 1 | data) = {p_or_s2_informed:.3f} for S2 and {p_or_s3_informed:.3f} for S3")
print(f"Difference: {abs(p_or_s2 - p_or_s2_informed)*100:.1f} for S2 and {abs(p_or_s3 - p_or_s3_informed)*100:.1f} for S3 percentage points")

# =============================================================
# SECTION 6: Plot posterior distributions
# =============================================================
# Create a figure with 2 rows, 2 columns:
#   Row 1: S2 — posterior of p_s2_ad and p_s2_pl overlaid
#   Row 2: S3 — posterior of p_s3_ad and p_s3_pl overlaid
fig, ax = plt.subplots(2,2, figsize =(8,8))

#plot season 2 seroconversion rate
samples_s2_ad = trace_uninformative.posterior["p_s2_ad"].values.flatten() 
samples_s2_pl = trace_uninformative.posterior["p_s2_pl"].values.flatten() 
ax[0, 0].hist(samples_s2_ad, bins=50, density=True, color='blue', alpha=0.6, label='A&D')
ax[0, 0].hist(samples_s2_pl, bins=50, density=True, color='green', alpha=0.6, label='Placebo')
ax[0, 0].set_title("Season 2 — Seroconversion Rate")
ax[0, 0].set_xlabel("Probability of seroconversion")
ax[0, 0].set_ylabel("Density")
ax[0,0].legend()

#plot season 3 seroconversion rate
samples_s3_ad = trace_uninformative.posterior["p_s3_ad"].values.flatten() 
samples_s3_pl = trace_uninformative.posterior["p_s3_pl"].values.flatten() 
ax[1, 0].hist(samples_s3_ad, bins=50, density=True, color='blue', alpha=0.6, label='A&D')
ax[1, 0].hist(samples_s3_pl, bins=50, density=True, color='green', alpha=0.6, label='Placebo')
ax[1, 0].set_title("Season 3 — Seroconversion Rate")
ax[1, 0].set_xlabel("Probability of seroconversion")
ax[1, 0].set_ylabel("Density")
ax[1,0].legend()

#plot season 2 OR
samples_or_s2 = trace_uninformative.posterior["or_s2"].values.flatten()
ax[0, 1].hist(samples_or_s2, bins=50, density=True, color='purple', alpha=0.6)
ax[0, 1].axvline(x=1.0, color='red', linestyle='--', label='OR=1 (no effect)') 
ax[0, 1].set_title("Season 2 — Odds Ratio posterior")
ax[0, 1].set_xlabel("Odds Ratio (A&D vs Placebo)")
ax[0, 1].set_ylabel("Density")
ax[0,1].legend()


#plot season 3 OR
samples_or_s3 = trace_uninformative.posterior["or_s3"].values.flatten()
ax[1, 1].hist(samples_or_s3, bins=50, density=True, color='purple', alpha=0.6)
ax[1, 1].axvline(x=1.0, color='red', linestyle='--', label='OR=1 (no effect)') 
ax[1, 1].set_title("Season 3 — Odds Ratio posterior")
ax[1, 1].set_xlabel("Odds Ratio (A&D vs Placebo)")
ax[1, 1].set_ylabel("Density")
ax[1,1].legend()

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig('../figures/03_bayesian_model.png', dpi=300, bbox_inches='tight')
print("Saved: ../figures/03_bayesian_model.png")
plt.show()


# =============================================================
# SECTION 7: Forest plot of posterior ORs
# =============================================================
# S2 uninformative
or_s2_mean_u = trace_uninformative.posterior["or_s2"].values.mean()
or_s2_hdi_u = az.hdi(trace_uninformative.posterior["or_s2"], hdi_prob=0.95)["or_s2"].values
p_s2_u = (trace_uninformative.posterior["or_s2"].values < 1.0).mean()

# S2 informed
or_s2_mean_i = trace_informative.posterior["or_s2"].values.mean()
or_s2_hdi_i  = az.hdi(trace_informative.posterior["or_s2"], hdi_prob=0.95)["or_s2"].values
p_s2_i = (trace_informative.posterior["or_s2"].values < 1.0).mean()

# S3 uninformative
or_s3_mean_u = trace_uninformative.posterior["or_s3"].values.mean()
or_s3_hdi_u = az.hdi(trace_uninformative.posterior["or_s3"], hdi_prob=0.95)["or_s3"].values
p_s3_u = (trace_uninformative.posterior["or_s3"].values < 1.0).mean()

# S3 informed
or_s3_mean_i = trace_informative.posterior["or_s3"].values.mean()
or_s3_hdi_i  = az.hdi(trace_informative.posterior["or_s3"], hdi_prob=0.95)["or_s3"].values
p_s3_i = (trace_informative.posterior["or_s3"].values < 1.0).mean()

rows = [
    (or_s2_mean_u, or_s2_hdi_u, p_s2_u),
    (or_s2_mean_i, or_s2_hdi_i, p_s2_i),
    (or_s3_mean_u, or_s3_hdi_u, p_s3_u),
    (or_s3_mean_i, or_s3_hdi_i, p_s3_i),
  ]

labels = ["S2 — Uninformative Beta(1,1)", 
          "S2 — Informed Beta(8,2)", 
          "S3 — Uninformative Beta(1,1)", 
          "S3 — Informed Beta(8,2)"]

colors = ['steelblue', 'lightblue', 'seagreen', 'lightgreen']

y_positions = [4, 3, 2, 1]

fig, ax = plt.subplots(figsize=(10, 5))

for i, (mean, hdi, p) in enumerate(rows):
    y = y_positions[i]
    ax.plot([hdi[0], hdi[1]], [y, y], color=colors[i], linewidth=2)
    ax.plot(mean, y, marker='s', color=colors[i], markersize=10)
    ax.text(hdi[1] + 0.05, y, f"OR={mean:.2f}  P(OR<1)={p:.1%}", va='center', fontsize=9)

ax.axvline(x=1.0, color='red', linestyle='--', label='OR=1 (no effect)')
ax.set_yticks(y_positions)
ax.set_yticklabels(labels)
ax.set_xlabel("Odds Ratio (A&D vs Placebo)")
ax.set_title("Forest Plot — Posterior ORs by Season and Prior")
ax.legend()
plt.tight_layout()
plt.savefig('../figures/03_bayesian_model_forest_plot.png', dpi=300, bbox_inches='tight')
print("Saved: ../figures/03_bayesian_model_forest_plot.png")
plt.show()



# =============================================================
# SECTION 8: Print "What the model tells us" summary
# =============================================================
# For each season print:
print(f"""Frequentist result 
    Season S2:
    OR = 0.13, 95% CI = [0.01, 1.21]
    CI crosses 1.0: True → NOT significant
    Wide CI = high uncertainty — not significant ≠ no effect
    Bayesian result with beta(1,1):
    OR = {or_s2_mean_u:.2f}, CI = [{or_s2_hdi_u[0]:.2f}, {or_s2_hdi_u[1]:.2f}] 
    Bayesian result with beta(8,2):
    OR = {or_s2_mean_i:.2f}, CI = [{or_s2_hdi_i[0]:.2f}, {or_s2_hdi_i[1]:.2f}] 
    
    Season S3:
    OR = 0.44, 95% CI = [0.06, 3.29]
    CI crosses 1.0: True → NOT significant
    Wide CI = high uncertainty — not significant ≠ no effect
    Bayesian result with beta(1,1):
    OR = {or_s3_mean_u:.2f}, CI = [{or_s3_hdi_u[0]:.2f}, {or_s3_hdi_u[1]:.2f}]
    Bayesian result with beta(8,2):
    OR = {or_s3_mean_i:.2f}, CI = [{or_s3_hdi_i[0]:.2f}, {or_s3_hdi_i[1]:.2f}] """)

# End with a comparison statement:
print(f"""
  CONCLUSION:
  Frequentist: p > 0.30 for both seasons — 'not significant'

  Bayesian:
    S2 — {p_s2_u:.1%} probability A&D reduces seroconversion (robust to prior choice)
    S3 — {p_s3_u:.1%} probability A&D reduces seroconversion (more uncertain)

  With small samples (n=11-21 per group), the data is consistent with A&D
  attenuating vaccine response — but also consistent with no effect. A larger trial is warranted.
  """)
