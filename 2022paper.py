# =============================================================================
# Steps 5–7: Paper 2 replication
#   Step 5 — Static vs Dynamic error
#   Step 6 — Cumulative error accumulation curves
#   Step 7 — Dynamic Time Warping (DTW) between days
#
# Assumes fem_temp_clean, male_temp_clean already exist (from your analysis.py).
# If running standalone, the data loading + cleaning block below handles it.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, ranksums

# ── Load and clean data (skip if already done in analysis.py) ───────
def clean_temp(df):
    df = df.copy()
    df[df < 35] = 35
    mean = df.mean()
    std  = df.std()
    df = df.clip(lower=mean - 3*std, upper=mean + 3*std, axis=1)
    return df

fem_temp  = pd.read_csv('FemTemp.csv',  index_col='time (min)')
male_temp = pd.read_csv('MaleTemp.csv', index_col='time (min)')
fem_clean  = clean_temp(fem_temp)
male_clean = clean_temp(male_temp)

MINS_PER_DAY = 1440
n_days = len(fem_clean) // MINS_PER_DAY


# =============================================================================
# STEP 5 — Static vs Dynamic error
# =============================================================================
#
# For every measurement, we ask: how far is this value from "normal"?
#
#   STATIC  baseline = one grand mean + SD pooled across ALL individuals
#                      and ALL timepoints. Treats all variance as random.
#
#   DYNAMIC baseline = a mean + SD computed per minute across all individuals.
#                      Each minute of the day has its own baseline, so
#                      circadian structure is already accounted for.
#
#   Error formula (from paper):
#       error = | (measurement − mean) / SD | − 1,  floored at 0
#
#   So a value within 1 SD of the mean contributes 0 error.
#   A value 3 SD away contributes error = 2.

print("=" * 60)
print("STEP 5: Static vs Dynamic Error")
print("=" * 60)

# Pool all individuals (male + female) across all timepoints
all_data = pd.concat([fem_clean, male_clean], axis=1)

# Static baseline: single scalar mean and SD
static_mean = all_data.values.mean()
static_sd   = all_data.values.std()
print(f"Static baseline — mean: {static_mean:.4f}°C, SD: {static_sd:.4f}°C")

# Dynamic baseline: one mean + SD per minute (Series of length n_timepoints)
dynamic_mean = all_data.mean(axis=1)   # shape: (n_timepoints,)
dynamic_sd   = all_data.std(axis=1)

def compute_error(df, mean, sd):
    """
    Pointwise error for every value in df.
    mean / sd can be a scalar (static) or a Series aligned to df.index (dynamic).
    Returns DataFrame same shape as df, values >= 0.
    """
    z     = (df.subtract(mean, axis=0)).divide(sd, axis=0)
    error = z.abs() - 1
    return error.clip(lower=0)

fem_static_err   = compute_error(fem_clean,  static_mean,  static_sd)
male_static_err  = compute_error(male_clean, static_mean,  static_sd)
fem_dynamic_err  = compute_error(fem_clean,  dynamic_mean, dynamic_sd)
male_dynamic_err = compute_error(male_clean, dynamic_mean, dynamic_sd)

print(f"\nMean static error  — females: {fem_static_err.values.mean():.4f} | "
      f"males: {male_static_err.values.mean():.4f}")
print(f"Mean dynamic error — females: {fem_dynamic_err.values.mean():.4f} | "
      f"males: {male_dynamic_err.values.mean():.4f}")

# Quick Kruskal-Wallis sanity check
stat, p = kruskal(male_static_err.values.flatten(), fem_static_err.values.flatten())
print(f"\nKruskal-Wallis static error  (M vs F): χ²={stat:.1f}, p={p:.2e}")
stat, p = kruskal(male_dynamic_err.values.flatten(), fem_dynamic_err.values.flatten())
print(f"Kruskal-Wallis dynamic error (M vs F): χ²={stat:.1f}, p={p:.2e}")

# Plot: each individual mouse as a line (like Paper 2 Fig 2A–D)
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
n_show = 2000  # show first 2000 minutes for readability

for mouse in fem_clean.columns:
    axes[0, 0].plot(fem_static_err[mouse].values[:n_show],
                    color='steelblue', alpha=0.4, linewidth=0.5)
for mouse in male_clean.columns:
    axes[0, 1].plot(male_static_err[mouse].values[:n_show],
                    color='crimson', alpha=0.4, linewidth=0.5)
for mouse in fem_clean.columns:
    axes[1, 0].plot(fem_dynamic_err[mouse].values[:n_show],
                    color='steelblue', alpha=0.4, linewidth=0.5)
for mouse in male_clean.columns:
    axes[1, 1].plot(male_dynamic_err[mouse].values[:n_show],
                    color='crimson', alpha=0.4, linewidth=0.5)

# overlay mean line
for ax, data, col in [
    (axes[0,0], fem_static_err,   'blue'),
    (axes[0,1], male_static_err,  'red'),
    (axes[1,0], fem_dynamic_err,  'blue'),
    (axes[1,1], male_dynamic_err, 'red'),
]:
    ax.plot(data.mean(axis=1).values[:n_show], color=col, linewidth=1.5)

titles = ['Females — Static Error', 'Males — Static Error',
          'Females — Dynamic Error', 'Males — Dynamic Error']
for ax, t in zip(axes.flatten(), titles):
    ax.set_title(t)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Error (SD units)')
    ax.set_ylim(0, 4)

plt.suptitle('Step 5: Static vs Dynamic Error per Individual\n(thick line = mean)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('step5_error.png', dpi=150, bbox_inches='tight')
print("\nSaved: step5_error.png")


# =============================================================================
# STEP 6 — Cumulative error accumulation curves
# =============================================================================
#
# Sum up error over time for each individual mouse. If males accumulate error
# faster, their cumulative curves will rise more steeply than females'.
#
# The GAP between the static and dynamic cumulative curves tells you how much
# variance is "structured" (accounted for by time of day).
# A bigger gap = more circadian structure in the data.
#
# Statistical test: Kruskal-Wallis on the LAST 1 hour of cumulative scores
# (i.e., the final total), comparing males vs females. (Paper 2, Fig 2E–F)

print("\n" + "=" * 60)
print("STEP 6: Cumulative Error Accumulation")
print("=" * 60)

def cumulative_error(error_df):
    """
    Cumulative sum of error over time for each mouse.
    Returns DataFrame same shape as error_df.
    """
    return error_df.cumsum(axis=0)

fem_static_cum   = cumulative_error(fem_static_err)
male_static_cum  = cumulative_error(male_static_err)
fem_dynamic_cum  = cumulative_error(fem_dynamic_err)
male_dynamic_cum = cumulative_error(male_dynamic_err)

# Statistical test on the last 60-minute window (last 1h of cumulative scores)
last_60 = 60
fem_static_final   = fem_static_cum.iloc[-last_60:].mean()    # mean of last 1h per mouse
male_static_final  = male_static_cum.iloc[-last_60:].mean()
fem_dynamic_final  = fem_dynamic_cum.iloc[-last_60:].mean()
male_dynamic_final = male_dynamic_cum.iloc[-last_60:].mean()

stat, p = kruskal(male_static_final.values, fem_static_final.values)
print(f"Kruskal-Wallis final static cumulative error  (M vs F): χ²={stat:.1f}, p={p:.2e}")
stat, p = kruskal(male_dynamic_final.values, fem_dynamic_final.values)
print(f"Kruskal-Wallis final dynamic cumulative error (M vs F): χ²={stat:.1f}, p={p:.2e}")

# How much does dynamic baseline reduce error vs static? (the ~30% the paper mentions)
fem_reduction  = 1 - (fem_dynamic_final.mean()  / fem_static_final.mean())
male_reduction = 1 - (male_dynamic_final.mean() / male_static_final.mean())
print(f"\nError reduction from dynamic baseline — females: {fem_reduction*100:.1f}%, "
      f"males: {male_reduction*100:.1f}%")

# Plot cumulative curves (like Paper 2 Fig 2E–F)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
t = np.arange(len(fem_static_cum))

for ax, title in zip(axes, ['CBT Cumulative Error', 'CBT Cumulative Error (zoomed)']):
    # individual mice (thin lines)
    for mouse in fem_clean.columns:
        ax.plot(t, fem_static_cum[mouse].values,
                color='steelblue', alpha=0.2, linewidth=0.5)
        ax.plot(t, fem_dynamic_cum[mouse].values,
                color='steelblue', alpha=0.2, linewidth=0.5, linestyle='--')
    for mouse in male_clean.columns:
        ax.plot(t, male_static_cum[mouse].values,
                color='crimson', alpha=0.2, linewidth=0.5)
        ax.plot(t, male_dynamic_cum[mouse].values,
                color='crimson', alpha=0.2, linewidth=0.5, linestyle='--')

    # sex means (thick lines)
    axes[0].plot(t, fem_static_cum.mean(axis=1).values,
                 color='blue', linewidth=2, label='Female static')
    axes[0].plot(t, fem_dynamic_cum.mean(axis=1).values,
                 color='blue', linewidth=2, linestyle='--', label='Female dynamic')
    axes[0].plot(t, male_static_cum.mean(axis=1).values,
                 color='red', linewidth=2, label='Male static')
    axes[0].plot(t, male_dynamic_cum.mean(axis=1).values,
                 color='red', linewidth=2, linestyle='--', label='Male dynamic')
    break  # only need to add legend once

axes[0].legend(fontsize=9)
axes[0].set_xlabel('Time (min)')
axes[0].set_ylabel('Cumulative Error (SD units)')
axes[0].set_title('Cumulative Error over Time\n(solid=static, dashed=dynamic)')

# Panel B: show reduction bracket like paper
axes[1].plot(t, male_static_cum.mean(axis=1).values,
             color='red', linewidth=2, linestyle='-',  label='Male static')
axes[1].plot(t, male_dynamic_cum.mean(axis=1).values,
             color='red', linewidth=2, linestyle='--', label='Male dynamic')
axes[1].plot(t, fem_static_cum.mean(axis=1).values,
             color='blue', linewidth=2, linestyle='-',  label='Female static')
axes[1].plot(t, fem_dynamic_cum.mean(axis=1).values,
             color='blue', linewidth=2, linestyle='--', label='Female dynamic')

# annotate reduction percentages
end_t = len(t) - 1
for color, static_val, dynamic_val, sex in [
    ('red',  male_static_final.mean(),  male_dynamic_final.mean(),  'Male'),
    ('blue', fem_static_final.mean(),   fem_dynamic_final.mean(),   'Female'),
]:
    reduction_pct = (1 - dynamic_val / static_val) * 100
    axes[1].annotate(f'{reduction_pct:.0f}% reduction',
                     xy=(end_t, (static_val + dynamic_val) / 2),
                     color=color, fontsize=9, ha='right')

axes[1].legend(fontsize=9)
axes[1].set_xlabel('Time (min)')
axes[1].set_ylabel('Cumulative Error (SD units)')
axes[1].set_title('Sex mean curves with reduction annotation')

plt.suptitle('Step 6: Cumulative Error Accumulation (Paper 2 Fig 2)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('step6_cumulative_error.png', dpi=150, bbox_inches='tight')
print("Saved: step6_cumulative_error.png")


# =============================================================================
# STEP 7 — Dynamic Time Warping (DTW) between consecutive days
# =============================================================================
#
# For each mouse, we compare every pair of consecutive days using DTW.
# DTW "warps" one day's time series to best align with the next,
# and returns the total distance needed — smaller distance = more self-similar.
#
# If females are more self-similar day-to-day, their DTW distances should
# be lower than males. (Paper 2, Fig 5)
#
# We use 5-minute bins (downsample by 5x) to keep runtime reasonable —
# this preserves ultradian and circadian structure while cutting 1440 → 288
# points per day.

print("\n" + "=" * 60)
print("STEP 7: Dynamic Time Warping (DTW)")
print("=" * 60)

def dtw_distance(s1, s2):
    """
    Pure numpy DTW with no boundary constraint.
    Matches MATLAB's dtw() with no 'sakoechiba' window limit.
    """
    n, m = len(s1), len(s2)
    cost_matrix = np.full((n + 1, m + 1), np.inf)
    cost_matrix[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i-1] - s2[j-1])
            cost_matrix[i, j] = cost + min(
                cost_matrix[i-1, j],      # insertion
                cost_matrix[i, j-1],      # deletion
                cost_matrix[i-1, j-1]     # match
            )
    return cost_matrix[n, m]

def compute_dtw_distances(df, downsample=5):
    """
    For each mouse, compute DTW distance between every pair of consecutive days.
    Returns dict: {mouse: [dist_day0->day1, dist_day1->day2, ...]}
    """
    n_days_local = len(df) // MINS_PER_DAY
    day_len = MINS_PER_DAY // downsample
    distances = {}

    for mouse in df.columns:
        signal = df[mouse].values[::downsample]   # downsample
        mouse_dists = []
        for day in range(n_days_local - 1):
            d1 = signal[day * day_len       : (day + 1) * day_len]
            d2 = signal[(day + 1) * day_len : (day + 2) * day_len]
            mouse_dists.append(dtw_distance(d1, d2))
        distances[mouse] = mouse_dists
        print(f"  {mouse}: mean DTW = {np.mean(mouse_dists):.2f}")

    return distances

print("\nComputing DTW for female mice...")
fem_dtw  = compute_dtw_distances(fem_clean)
print("\nComputing DTW for male mice...")
male_dtw = compute_dtw_distances(male_clean)

# Flatten all days x all mice
fem_dtw_flat  = [d for dists in fem_dtw.values()  for d in dists]
male_dtw_flat = [d for dists in male_dtw.values() for d in dists]

print(f"\nMean DTW distance — females: {np.mean(fem_dtw_flat):.2f} | "
      f"males: {np.mean(male_dtw_flat):.2f}")

stat, p = ranksums(male_dtw_flat, fem_dtw_flat)
print(f"Wilcoxon rank sum (M vs F) raw DTW: p={p:.2e}")

# SD-corrected DTW (divide each mouse's distances by its own SD)
# This checks if the difference holds after normalizing for overall variance
fem_dtw_corrected  = []
male_dtw_corrected = []

for mouse, dists in fem_dtw.items():
    sd = fem_clean[mouse].std()
    fem_dtw_corrected.extend([d / sd for d in dists])

for mouse, dists in male_dtw.items():
    sd = male_clean[mouse].std()
    male_dtw_corrected.extend([d / sd for d in dists])

stat, p = ranksums(male_dtw_corrected, fem_dtw_corrected)
print(f"Wilcoxon rank sum (M vs F) SD-corrected DTW: p={p:.2e}")
print(f"Mean SD-corrected DTW — females: {np.mean(fem_dtw_corrected):.2f} | "
      f"males: {np.mean(male_dtw_corrected):.2f}")

# Plot (like Paper 2 Fig 5)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Per-mouse mean DTW as boxplot
fem_mouse_means  = [np.mean(v) for v in fem_dtw.values()]
male_mouse_means = [np.mean(v) for v in male_dtw.values()]

fem_mouse_means_corr  = []
male_mouse_means_corr = []
for mouse, dists in fem_dtw.items():
    sd = fem_clean[mouse].std()
    fem_mouse_means_corr.append(np.mean([d/sd for d in dists]))
for mouse, dists in male_dtw.items():
    sd = male_clean[mouse].std()
    male_mouse_means_corr.append(np.mean([d/sd for d in dists]))

bp1 = axes[0].boxplot([male_mouse_means, fem_mouse_means],
                       labels=['Males', 'Females'],
                       patch_artist=True)
bp1['boxes'][0].set_facecolor('mistyrose')
bp1['boxes'][1].set_facecolor('lightblue')
axes[0].set_title('Raw DTW Distance (day-to-day)')
axes[0].set_ylabel('DTW distance (°C)')

bp2 = axes[1].boxplot([male_mouse_means_corr, fem_mouse_means_corr],
                       labels=['Males', 'Females'],
                       patch_artist=True)
bp2['boxes'][0].set_facecolor('mistyrose')
bp2['boxes'][1].set_facecolor('lightblue')
axes[1].set_title('SD-corrected DTW Distance')
axes[1].set_ylabel('DTW distance / individual SD')

for ax in axes:
    ax.set_xlabel('Sex')

plt.suptitle('Step 7: DTW Day-to-Day Similarity (Paper 2 Fig 5)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('step7_dtw.png', dpi=150, bbox_inches='tight')
print("\nSaved: step7_dtw.png")
print("\nAll done! ✓")