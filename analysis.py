# analysis.py
import pandas as pd
import matplotlib as plt

# 1. Load data (FemTemp.csv, MaleTemp.csv, FemAct.csv, MaleAct.csv)
fem_temp = pd.read_csv('FemTemp.csv', index_col='time (min)')
male_temp = pd.read_csv('MaleTemp.csv', index_col='time (min)')
fem_act  = pd.read_csv('FemAct.csv',  index_col='time (min)')
male_act = pd.read_csv('MaleAct.csv', index_col='time (min)')

for name, df in [('FemTemp', fem_temp), ('MaleTemp', male_temp), ('FemAct',  fem_act),  ('MaleAct',  male_act)]:
    print(f"{name}: {df.shape} | days: {len(df)/1440:.1f} | mice: {df.shape[1]}")


# 2. Data cleaning (clip outliers, handle missing values)
def clean_temp(df):
    df = df.copy()          # make a copy so we don't modify the original
    df[df < 35] = 35        # any temperature below 35°C gets set TO 35
                            # (paper says these are device malfunctions)
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
                                        # (mouse is sleeping!)
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

# 5. Static vs dynamic error (Paper 2)

# 6. Cumulative error accumulation curves (Paper 2)

# 7. Dynamic time warping between days (Paper 2)

# 8. Plots