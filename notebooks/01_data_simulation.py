import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

#set random seed
np.random.seed(42)

# #set plot style
# sns.set_style('whitegrid')
# plt.rcParams['figure.figsize']=(10,6)

#season 2 (2016-2017) parameters
s2_ad_n = 21
s2_ad_seroconverted = 15
s2_placebo_n = 20
s2_placebo_seroconverted = 19 

#season 3 (2017-2018) parameters
s3_ad_n = 11
s3_ad_seroconverted = 8
s3_placebo_n = 14
s3_placebo_seroconverted = 12

print("Season 2 - A&D group: {}/{} = {:.1%}".format(s2_ad_seroconverted, s2_ad_n, s2_ad_seroconverted/s2_ad_n))
print("Season 2 - Placebo: {}/{} = {:.1%}".format(s2_placebo_seroconverted, s2_placebo_n, s2_placebo_seroconverted/s2_placebo_n))
print("\nSeason 3 - A&D group: {}/{} = {:.1%}".format(s3_ad_seroconverted, s3_ad_n, s3_ad_seroconverted/s3_ad_n))
print("Season 3 - Placebo: {}/{} = {:.1%}".format(s3_placebo_seroconverted, s3_placebo_n, s3_placebo_seroconverted/s3_placebo_n))

#simulate patient data
def simulate_group(season, treatment, n_total, n_seroconverted):
    #create seroconversion outcomes: 1=yes, 0=no
    outcomes = np.array([1] * n_seroconverted + [0] * (n_total-n_seroconverted))

    #shuffle to randomize order
    np.random.shuffle(outcomes)

    #create dataframe 
    df = pd.DataFrame({
        'participant_id': range(1, n_total+1),
        'season':season,
        'treatment':treatment,
        'seroconverted':outcomes
    })

    return df

#simulate treatment and placebo for both seasons: 4 groups
s2_ad = simulate_group('S2', 'A&D', s2_ad_n, s2_ad_seroconverted)
s2_placebo = simulate_group('S2','Placebo', s2_placebo_n, s2_placebo_seroconverted)
s3_ad = simulate_group('S3', 'A&D', s3_ad_n, s3_ad_seroconverted)
s3_placebo = simulate_group('S3','Placebo', s3_placebo_n, s3_placebo_seroconverted)

#combine seasons in single dataset
df = pd.concat([s2_ad, s2_placebo, s3_ad, s3_placebo], ignore_index=True)

#overwrite per-group IDs with unique IDs across full dataset
df['participant_id'] = range(1, len(df) +1)

print(f"Total participants simulated: {len(df)}")
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head(2))

#verify simulated data matches summary statistics
summary = df.groupby(['season','treatment']).agg({
    'seroconverted': ['count','sum','mean']
}).round(3)

summary.columns = ['n_total','n_seroconverted','rate']
summary['percentage'] = (summary['rate']*100).round(1).astype(str)+'%'

print("\n=== SIMULATED DATA SUMMARY ===")
print(summary)

print("\n=== COMPARISON TO REPORTED STATISTICS ===")
print("Season 2 A&D: Simulated vs Reported")
print(f"  {summary.loc[('S2', 'A&D'), 'percentage']} vs 71.4%")
print("\nSeason 2 Placebo: Simulated vs Reported")
print(f"  {summary.loc[('S2', 'Placebo'), 'percentage']} vs 95.0%")
print("\nSeason 3 A&D: Simulated vs Reported")
print(f"  {summary.loc[('S3', 'A&D'), 'percentage']} vs 72.7%")
print("\nSeason 3 Placebo: Simulated vs Reported")
print(f"  {summary.loc[('S3', 'Placebo'), 'percentage']} vs 85.7%")


#visualisation 

#calculate rates
plot_data = df.groupby(['season','treatment'], as_index = False).agg(rate=('seroconverted','mean'))
plot_data['rate_pct']=plot_data['rate']*100

#create bar plot 
fig, ax = plt.subplots(figsize=(10,6))
x = np.arange(2) #two seasons
width=0.35
colors= ['#FF6B6B', '#4ECDC4']

#plot bars
sns.barplot(
    data=plot_data,
    x='season',
    y='rate_pct',
    hue='treatment',
    ax=ax
)

# Add percentage labels on top
for p in ax.patches:
    ax.text(p.get_x()+p.get_width()/2, p.get_height()+1, f'{p.get_height():.1f}%', 
            ha='center', va='bottom')

plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('../figures/seroconversion_by_group.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFigure saved to: figures/seroconversion_by_group.png")

def calculate_odds_ratio(treatment_successes, treatment_total, placebo_successes, placebo_total):
    """
    Calculate odds ratio for treatment vs placebo.
    
    OR > 1: Treatment increases odds of outcome
    OR < 1: Treatment decreases odds of outcome
    OR = 1: No effect
    """
    # Calculate odds for each group
    treatment_failures = treatment_total - treatment_successes
    placebo_failures = placebo_total - placebo_successes
    
    treatment_odds = treatment_successes / treatment_failures if treatment_failures > 0 else float('inf')
    placebo_odds = placebo_successes / placebo_failures if placebo_failures > 0 else float('inf')
    
    # Calculate OR
    odds_ratio = treatment_odds / placebo_odds
    
    return odds_ratio, treatment_odds, placebo_odds

# Season 2
s2_or, s2_treat_odds, s2_plac_odds = calculate_odds_ratio(
    s2_ad_seroconverted, s2_ad_n, 
    s2_placebo_seroconverted, s2_placebo_n
)

# Season 3
s3_or, s3_treat_odds, s3_plac_odds = calculate_odds_ratio(
    s3_ad_seroconverted, s3_ad_n,
    s3_placebo_seroconverted, s3_placebo_n
)

print("\n=== ODDS RATIOS (A&D vs Placebo) ===")
print(f"\nSeason 2:")
print(f"  A&D odds: {s2_treat_odds:.2f}")
print(f"  Placebo odds: {s2_plac_odds:.2f}")
print(f"  Odds Ratio: {s2_or:.2f}")
print(f"  Interpretation: A&D has {s2_or:.2f}x the odds of seroconversion vs Placebo")
print(f"  {'A&D REDUCES SEROCONVERSION (OR < 1)' if s2_or < 1 else 'A&D INCREASES SEROCONVERSION (OR > 1)' if s2_or > 1 else 'NO EFFECT'}")

print(f"\nSeason 3:")
print(f"  A&D odds: {s3_treat_odds:.2f}")
print(f"  Placebo odds: {s3_plac_odds:.2f}")
print(f"  Odds Ratio: {s3_or:.2f}")
print(f"  Interpretation: A&D has {s3_or:.2f}x the odds of seroconversion vs Placebo")
print(f"  {'A&D REDUCES SEROCONVERSION (OR < 1)' if s3_or < 1 else 'A&D INCREASES SEROCONVERSION (OR > 1)' if s3_or > 1 else 'NO EFFECT'}")

print("\n⚠️ NOTE: OR < 1.0 means A&D supplementation is associated with LOWER seroconversion!")
print("This is opposite of expected (vitamin A should improve immune response).")
print("This is why Bayesian analysis is valuable - quantifying effect size + uncertainty.")


# Save to CSV
output_path = '../data/simulated_patient_data.csv'
df.to_csv(output_path, index=False)

print(f"\nSimulated data saved to: {output_path}")
print(f"Total participants: {len(df)}")
print(f"\nDataset preview:")
print(df.head(10))