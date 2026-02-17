# ============================================================
# 02 - Exploratory Analysis — EXERCISE GUIDE
# Write your code below each instruction
# Goal: Show WHY the frequentist analysis is insufficient
#       and motivate the Bayesian approach
# ============================================================

# ---- SETUP: imports you will need ----
# pandas, numpy, matplotlib.pyplot, seaborn, scipy.stats
# Write them here:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats 
from statsmodels.stats.proportion import proportion_confint

# =============================================================
# SECTION 1: Load the data
# =============================================================

df = pd.read_csv("../data/simulated_patient_data.csv")

# =============================================================
# SECTION 2: Seroconversion rates per group
# =============================================================

summary = df.groupby(['season','treatment']).agg(n_total=('participant_id','count'),n_seroconverted=('seroconverted','sum')).reset_index()
summary['rate'] =summary['n_seroconverted']/summary['n_total']

# =============================================================
# SECTION 3: Add 95% Confidence Intervals
# 
def wilson_ci(n_success, n_total):
      lower, upper = proportion_confint(n_success, n_total, alpha=0.05, method='wilson')
      return lower, upper

for idx, row in summary.iterrows():
    lower, upper = wilson_ci(row['n_seroconverted'], row['n_total'])
    summary.loc[idx, 'ci_lower'] = lower
    summary.loc[idx, 'ci_upper'] = upper


# =============================================================
# SECTION 4: Bar chart with error bars
# =============================================================
# Create a figure with 1 row, 2 columns (one subplot per season: S2, S3)
# For each season:
#   - Plot a bar for A&D and a bar for Placebo (use rate_pct)
#   - Add error bars using ax.errorbar() — asymmetric (distance to CI bounds)
#   - Add percentage labels on top of each bar
#   - Set title, ylabel, xticks, ylim(0, 115)
#
# Save to '../figures/02_seroconversion_with_ci.png'
# Print confirmation message
summary['rate_pct'] = summary['rate']*100
summary['ci_lower_pct'] = summary['ci_lower']*100
summary['ci_upper_pct'] = summary['ci_upper']*100

print(summary)

fig, axes = plt.subplots(1, 2, figsize=(12,6))
for ax, season in zip(axes, ['S2', 'S3']):
      season_data = summary[summary['season'] == season].reset_index(drop=True)

      # Bars
      ax.bar(np.arange(2), season_data['rate_pct'], width=0.5,
             color=['#FF6B6B', '#4ECDC4'])

      # Asymmetric error bars
      for i, row in season_data.iterrows():
          lower_err = row['rate_pct'] - row['ci_lower_pct']
          upper_err = row['ci_upper_pct'] - row['rate_pct']
          ax.errorbar(i, row['rate_pct'],
                      yerr=[[lower_err], [upper_err]],
                      fmt='none', color='black', capsize=5)

      # Labels on top of bars
      for i, row in season_data.iterrows():
          ax.text(i, row['rate_pct'] + 2, f"{row['rate_pct']:.1f}%",
                  ha='center', va='bottom')

      ax.set_title(f'Season {season}')
      ax.set_ylabel('Seroconversion Rate (%)')
      ax.set_xticks(np.arange(2))
      ax.set_xticklabels(season_data['treatment'])
      ax.set_ylim(0, 115)

  
plt.tight_layout()
plt.savefig('../figures/02_seroconversion_with_ci.png', dpi=300, bbox_inches='tight')
print("Saved: ../figures/02_seroconversion_with_ci.png")
plt.show()


# =============================================================
# SECTION 5: Fisher's exact test 
# =============================================================


for season in ['S2', 'S3']:    
      ad_row = summary[(summary['season'] == season) & (summary['treatment'] == 'A&D')].iloc[0]                                                             
      placebo_row = summary[(summary['season'] == season) & (summary['treatment'] == 'Placebo')].iloc[0]   
      ad_failures = ad_row['n_total'] - ad_row['n_seroconverted'] 
      placebo_failures = placebo_row['n_total'] - placebo_row['n_seroconverted'] 
      table = [[ad_row['n_seroconverted'], ad_failures],[placebo_row['n_seroconverted'], placebo_failures]]
      
      odds_ratio, p_value = scipy.stats.fisher_exact(table)

      print(f"Season {season}")                                                                                                                                 
      print(f"Table: {table}")
      print(f"Odds Ratio: {odds_ratio:.3f}")
      print(f"p-value: {p_value:.3f}")
      print(f"Significant: {p_value < 0.05}")
      print()



# =============================================================
# SECTION 6: Forest plot of Odds Ratios
# =============================================================
# 

def odds_ratio_ci(n1_success, n1_total, n2_success, n2_total):                                                                                            
      n1_fail = n1_total - n1_success                                            
      n2_fail = n2_total - n2_success                                                                                                                       
      OR = (n1_success / n1_fail) / (n2_success / n2_fail)                                                                                                  
      log_OR = np.log(OR)
      se = np.sqrt(1/n1_success + 1/n1_fail + 1/n2_success + 1/n2_fail)
      ci_lower = np.exp(log_OR - 1.96 * se)
      ci_upper = np.exp(log_OR + 1.96 * se)
      return OR, ci_lower, ci_upper

results = {}
for season in ['S2', 'S3']:
      ad = summary[(summary['season'] == season) & (summary['treatment'] == 'A&D')].iloc[0]
      pl = summary[(summary['season'] == season) & (summary['treatment'] == 'Placebo')].iloc[0]
      OR, ci_lower, ci_upper = odds_ratio_ci(ad['n_seroconverted'], ad['n_total'],
                                              pl['n_seroconverted'], pl['n_total'])
      results[season] = (OR, ci_lower, ci_upper)

fig, ax = plt.subplots(figsize=(8, 4))
for y, season in zip([2, 1], ['S2', 'S3']):
      OR, ci_lower, ci_upper = results[season]
      n = summary[summary['season'] == season]['n_total'].sum()
      ax.plot([ci_lower, ci_upper], [y, y], 'b-', linewidth=2)
      ax.plot(OR, y, 's', color='blue', markersize=10)
      ax.text(ci_upper + 0.05, y, f"OR={OR:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]", va='center')
      ax.text(0.05, y, f"{season} (n={n})", va='center', ha='left')

ax.axvline(1.0, color='red', linestyle='--', label='OR=1 (null)')
ax.set_yticks([])
ax.set_xlabel('Odds Ratio')
ax.set_title('Forest Plot — Observed Odds Ratios')
ax.legend()
plt.tight_layout()
plt.savefig('../figures/02_forest_plot_observed.png', dpi=300, bbox_inches='tight')
print("Saved: ../figures/02_forest_plot_observed.png")
plt.show()


# =============================================================
# SECTION 7: Print "Why Bayesian?" summary
# =============================================================
#print("=" * 60)                                                                                                                                           
print("WHY BAYESIAN? — Summary")                            
print("=" * 60)                                                                                                                                           
                                                                                                                                                            
for season in ['S2', 'S3']:                                                                                                                               
      OR, ci_lower, ci_upper = results[season]                                                                                                              
      crosses_null = ci_lower < 1.0 < ci_upper
      print(f"\nSeason {season}:")
      print(f"  OR = {OR:.2f}, 95% CI = [{ci_lower:.2f}, {ci_upper:.2f}]")
      print(f"  CI crosses 1.0: {crosses_null} → {'NOT significant' if crosses_null else 'Significant'}")
      print(f"  Wide CI = high uncertainty — 'not significant' ≠ 'no effect'")

print("""
  What the Bayesian model adds:
    - P(OR < 1 | data): direct probability that A&D reduces seroconversion
    - 95% Credible Interval: where the true OR likely lies, given the data
    - Prior sensitivity analysis: does the conclusion change with different priors?

  Frequentist verdict: 'not significant' (p > 0.30)
  Bayesian question:   'what is the probability there is an effect?'
  """)