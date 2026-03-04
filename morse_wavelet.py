# morse_wavelet.py
# Drop-in replacement for Step 4 in analysis.py
# Uses Morse wavelet (beta=5, gamma=3) matching the paper exactly,
# instead of pywt's cmor which is a different wavelet family.

import numpy as np
import pandas as pd
import os
import time


def morse_wavelet_cwt(signal, periods_minutes, beta=5, gamma=3, fs=1.0):
    """
    Continuous Wavelet Transform using the Morse wavelet.

    This matches the Morse wavelet used in the Smarr & Kriegsfeld papers
    (beta=5, gamma=3), implemented via convolution in the frequency domain.

    Parameters
    ----------
    signal          : 1D numpy array, your time series (1 value per minute)
    periods_minutes : 1D array of periods to analyze, in minutes
                      e.g. np.arange(23*60, 25*60) for circadian band
    beta, gamma     : Morse wavelet shape parameters (paper uses 5 and 3)
    fs              : sampling frequency in samples/min (default 1.0)

    Returns
    -------
    power : 2D array of shape (len(periods_minutes), len(signal))
            power[i, t] = wavelet power at period i and time t
    """
    n          = len(signal)
    sig_fft    = np.fft.fft(signal, n=n)
    freqs      = np.fft.fftfreq(n, d=1.0 / fs)   # cycles per minute

    power = np.zeros((len(periods_minutes), n))

    for i, period in enumerate(periods_minutes):
        # convert period (minutes) → scale parameter
        scale = period / (2 * np.pi)
        omega = 2 * np.pi * freqs * scale          # dimensionless frequency

        # Morse wavelet in frequency domain (only defined for omega > 0)
        psi_fft      = np.zeros(n, dtype=complex)
        pos          = omega > 0
        psi_fft[pos] = (omega[pos] ** beta) * np.exp(-(omega[pos] ** gamma))

        # L2 normalisation
        psi_fft *= np.sqrt(2 * np.pi * scale)

        # wavelet coefficients via inverse FFT
        coeffs      = np.fft.ifft(sig_fft * np.conj(psi_fft))
        power[i, :] = np.abs(coeffs) ** 2

    return power   # shape: (n_periods, n_timepoints)


def compute_bandpower(df, band_min_h, band_max_h,
                      beta=5, gamma=3, step_min=10):
    """
    Apply morse_wavelet_cwt to every mouse column in df and extract
    the MAX power per minute within the specified period band.

    This matches the paper's definition:
        "max wavelet power per minute for the range of periodicities"

    Parameters
    ----------
    df           : DataFrame, rows = timepoints (minutes), cols = mice
    band_min_h   : lower bound of period band, in HOURS (e.g. 1 for ultradian)
    band_max_h   : upper bound of period band, in HOURS (e.g. 3 for ultradian)
    beta, gamma  : Morse parameters
    step_min     : period step size in minutes (coarser = faster, default 10)

    Returns
    -------
    DataFrame of shape (n_timepoints, n_mice) — max band power per minute
    """
    periods = np.arange(band_min_h * 60, band_max_h * 60, step_min)
    band_power = {}

    for mouse in df.columns:
        print(f"    {mouse}...", flush=True)
        signal = df[mouse].values
        power  = morse_wavelet_cwt(signal, periods, beta=beta, gamma=gamma)
        # max power across all periods in band, per timepoint
        band_power[mouse] = power.max(axis=0)

    return pd.DataFrame(band_power, index=df.index)


def compute_or_load(filepath, compute_fn, *args, **kwargs):
    """Load result from CSV cache if it exists, otherwise compute and save."""
    if os.path.exists(filepath):
        print(f"  Loading cached {filepath}...")
        return pd.read_csv(filepath, index_col=0)
    else:
        print(f"  Computing {filepath}...")
        t0     = time.time()
        result = compute_fn(*args, **kwargs)
        result.to_csv(filepath)
        print(f"  Done — took {(time.time()-t0)/60:.1f} min, saved to {filepath}")
        return result
