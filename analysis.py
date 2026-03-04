# analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import pywt
from morse_wavelet import compute_bandpower, compute_or_load


# 1. Load data 
fem_temp = pd.read_csv('FemTemp.csv', index_col='time (min)')
male_temp = pd.read_csv('MaleTemp.csv', index_col='time (min)')
fem_act  = pd.read_csv('FemAct.csv',  index_col='time (min)')
male_act = pd.read_csv('MaleAct.csv', index_col='time (min)')

for name, df in [('FemTemp', fem_temp), ('MaleTemp', male_temp), ('FemAct',  fem_act),  ('MaleAct',  male_act)]:
    print(f"{name}: {df.shape} | days: {len(df)/1440:.1f} | mice: {df.shape[1]}")


# 2. Data cleaning (clip outliers, handle missing values)
def clean_temp(df):
    df = df.copy()          # make a copy so we don't modify the original
    df[df < 35] = 35        # any temperature below 35°C gets set TO 35 as 
                            # it's because device malfunctions
    mean = df.mean()        # calculate average of each mouse's column
    std  = df.std()         # calculate standard deviation of each column
    df = df.clip(lower=mean - 3*std, upper=mean + 3*std, axis = 1)  
                            # anything more than 3 standard deviations 
                            # from the mean gets clipped to that boundary
    return df               # return the cleaned table

def clean_act(df):
    df = df.copy()
    mean = df.mean()
    std  = df.std()
    df = df.clip(upper=mean + 3*std, axis = 1)   # only clip HIGH values
                                        # we don't clip low values because
                                        # activity = 0 is totally normal
                                        #sleeeeeeeeeeeeeeeeeeeep
    return df

fem_temp_clean  = clean_temp(fem_temp)   # cleaned version of female temps
male_temp_clean = clean_temp(male_temp)  # cleaned version of male temps
fem_act_clean   = clean_act(fem_act)     # cleaned version of female activity
male_act_clean  = clean_act(male_act)    # cleaned version of male activity

# 3. Daily range analysis (Paper 1)
import matplotlib.pyplot as plt # type: ignore
from scipy.stats import ranksums   # type: ignore # for Wilcoxon rank sum test

# ── Step 3: Daily range analysis ───────────────────────────────────

MINS_PER_DAY = 1440  # 60 mins * 24 hours

def daily_range(df):
    """For each mouse, compute max-min for each day."""
    n_days = len(df) // MINS_PER_DAY
    results = []
    for day in range(n_days):
        start = day * MINS_PER_DAY
        end   = start + MINS_PER_DAY
        day_data = df.iloc[start:end]          # slice one day's worth of rows
        results.append(day_data.max() - day_data.min())  # range per mouse
    return pd.DataFrame(results)               # shape: (n_days, n_mice)

fem_temp_range  = daily_range(fem_temp_clean)
male_temp_range = daily_range(male_temp_clean)
fem_act_range   = daily_range(fem_act_clean)
male_act_range  = daily_range(male_act_clean)

# ── Statistical comparison (Wilcoxon rank sum test) ─────────────────
# flatten all days x all mice into one list per sex
stat, p = ranksums(male_temp_range.values.flatten(),
                   fem_temp_range.values.flatten())
print(f"CBT daily range — males: {male_temp_range.values.mean():.2f} | "
      f"females: {fem_temp_range.values.mean():.2f} | p={p:.4f}")

stat, p = ranksums(male_act_range.values.flatten(),
                   fem_act_range.values.flatten())
print(f"ACT daily range — males: {male_act_range.values.mean():.2f} | "
      f"females: {fem_act_range.values.mean():.2f} | p={p:.4f}")

# ── Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].boxplot([male_temp_range.values.flatten(),
                 fem_temp_range.values.flatten()],
                labels=['Males', 'Females'])
axes[0].set_title('CBT Daily Range')
axes[0].set_ylabel('°C')

axes[1].boxplot([male_act_range.values.flatten(),
                 fem_act_range.values.flatten()],
                labels=['Males', 'Females'])
axes[1].set_title('Activity Daily Range')
axes[1].set_ylabel('counts')

plt.tight_layout()
plt.savefig('daily_range.png')
print("Plot saved as daily_range.png!")
# 4. Wavelet transforms + band power extraction (Paper 1)
import pywt
import os
import time

# ── Step 4: Wavelet analysis ────────────────────────────────────────

TRIM_CIRC = 1440   # trim 1 day from each end for circadian band (edge effects)
TRIM_ULTR = 180    # trim 3 h  from each end for ultradian band

# ── Temperature ─────────────────────────────────────────────────────
print("Computing wavelet band power (temperature)...")

fem_temp_circ  = compute_or_load('fem_temp_circ.csv',  compute_bandpower,
                                  fem_temp_clean,  23, 25)
male_temp_circ = compute_or_load('male_temp_circ.csv', compute_bandpower,
                                  male_temp_clean, 23, 25)
fem_temp_ultr  = compute_or_load('fem_temp_ultr.csv',  compute_bandpower,
                                  fem_temp_clean,  1,  3)
male_temp_ultr = compute_or_load('male_temp_ultr.csv', compute_bandpower,
                                  male_temp_clean, 1,  3)

# ── Activity ─────────────────────────────────────────────────────────
print("Computing wavelet band power (activity)...")

fem_act_circ   = compute_or_load('fem_act_circ.csv',   compute_bandpower,
                                  fem_act_clean,   23, 25)
male_act_circ  = compute_or_load('male_act_circ.csv',  compute_bandpower,
                                  male_act_clean,  23, 25)
fem_act_ultr   = compute_or_load('fem_act_ultr.csv',   compute_bandpower,
                                  fem_act_clean,   1,  3)
male_act_ultr  = compute_or_load('male_act_ultr.csv',  compute_bandpower,
                                  male_act_clean,  1,  3)

# ── Trim edge effects ────────────────────────────────────────────────
fem_temp_circ_trim  = fem_temp_circ.iloc[TRIM_CIRC:-TRIM_CIRC]
male_temp_circ_trim = male_temp_circ.iloc[TRIM_CIRC:-TRIM_CIRC]
fem_temp_ultr_trim  = fem_temp_ultr.iloc[TRIM_ULTR:-TRIM_ULTR]
male_temp_ultr_trim = male_temp_ultr.iloc[TRIM_ULTR:-TRIM_ULTR]

fem_act_circ_trim   = fem_act_circ.iloc[TRIM_CIRC:-TRIM_CIRC]
male_act_circ_trim  = male_act_circ.iloc[TRIM_CIRC:-TRIM_CIRC]
fem_act_ultr_trim   = fem_act_ultr.iloc[TRIM_ULTR:-TRIM_ULTR]
male_act_ultr_trim  = male_act_ultr.iloc[TRIM_ULTR:-TRIM_ULTR]

# ── Statistical comparison (Wilcoxon rank sum, per-mouse medians) ────
from scipy.stats import ranksums

results = [
    ("Temp ultradian",  male_temp_ultr_trim, fem_temp_ultr_trim),
    ("Temp circadian",  male_temp_circ_trim, fem_temp_circ_trim),
    ("Act  ultradian",  male_act_ultr_trim,  fem_act_ultr_trim),
    ("Act  circadian",  male_act_circ_trim,  fem_act_circ_trim),
]

print("\nWilcoxon rank sum — per-mouse median band power (males vs females):")
for label, male_df, fem_df in results:
    m_med = male_df.median().values   # one median per mouse
    f_med = fem_df.median().values
    stat, p = ranksums(m_med, f_med)
    print(f"  {label:20s}  males: {m_med.mean():.4f}  females: {f_med.mean():.4f}  p={p:.4f}")

# ── Plot ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

plot_data = [
    (axes[0,0], male_temp_ultr_trim, fem_temp_ultr_trim, 'Temp Ultradian Power (1–3h)'),
    (axes[0,1], male_temp_circ_trim, fem_temp_circ_trim, 'Temp Circadian Power (23–25h)'),
    (axes[1,0], male_act_ultr_trim,  fem_act_ultr_trim,  'Activity Ultradian Power (1–3h)'),
    (axes[1,1], male_act_circ_trim,  fem_act_circ_trim,  'Activity Circadian Power (23–25h)'),
]

for ax, male_df, fem_df, title in plot_data:
    ax.plot(male_df.mean(axis=1).values, color='red',  label='Males',   linewidth=1.2)
    ax.plot(fem_df.mean(axis=1).values,  color='blue', label='Females', linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Max power')
    ax.legend(fontsize=8)

plt.suptitle('Morse Wavelet Band Power — Males vs Females\n(Paper 1 replication)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('wavelet_power_morse.png', dpi=150, bbox_inches='tight')
print("\nSaved: wavelet_power_morse.png")