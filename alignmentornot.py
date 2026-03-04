# analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums



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