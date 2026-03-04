import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('micetempact\FemTemp.csv')
mice = [c for c in df.columns if c != 'time (min)']

n_days = 8
mins_per_day = 1440

# Plot
fig, axes = plt.subplots(7, 2, figsize=(14, 20))
axes = axes.flatten()

for idx, mouse in enumerate(mice):
    ax = axes[idx]
    temps = df[mouse].values

    daily_max = []
    for day in range(n_days):
        day_data = temps[day*mins_per_day:(day+1)*mins_per_day]
        daily_max.append(np.nanmax(day_data))

    days = np.arange(1, n_days+1)
    bars = ax.bar(days, daily_max, color='steelblue', alpha=0.7, edgecolor='white')

    # Highlight the day with max temp in red
    peak_day = np.argmax(daily_max) + 1
    bars[peak_day-1].set_color('crimson')
    bars[peak_day-1].set_alpha(1.0)

    ax.set_title(f'{mouse} — peak day: {peak_day}', fontsize=11)
    ax.set_xlabel('Day')
    ax.set_ylabel('Max CBT (°C)')
    ax.set_xticks(days)
    ax.set_ylim(38, 40.5)

plt.suptitle('Daily Maximum CBT per Female Mouse\n(red = peak day)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fem_daily_max_temp.png', dpi=150, bbox_inches='tight')
plt.show()

# Print peak days to console
print("Peak days:")
for mouse in mice:
    temps = df[mouse].values
    daily_max = [np.nanmax(temps[d*mins_per_day:(d+1)*mins_per_day]) for d in range(n_days)]
    print(f"  {mouse}: day {np.argmax(daily_max)+1} ({max(daily_max):.2f}°C)")