import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 1. Load your dataset
df = pd.read_parquet('./data/cpsc_2018_labeled_single_lead.parquet')
# Print all unique values in the label column
print(df['label'].unique())
# Assuming you have a column 'label' or 'diagnosis' and 'x' is the signal
# Filter only for the AF recordings to see the AF distribution
df_af = df[df['label'] == 1].copy() 

# Sampling rate for CPSC 2018 is usually 500 Hz
fs = 100

def calculate_diffusivity(signal, fs=100, window_sec=5, cv_threshold=0.10):
    """
    Estimates the percentage of a signal spent in AF by looking at RR interval variability.
    """
    # Find R-peaks (simple thresholding - you may want to use neurokit2 or biosppy for better accuracy)
    # Distance of 0.4s assumes a max HR of 150 bpm for basic R-peak detection
    peaks, _ = find_peaks(signal, distance=fs*0.4, prominence=0.5)
    
    if len(peaks) < 2:
        return 0.0 # Not enough beats
        
    # Calculate RR intervals in seconds
    rr_intervals = np.diff(peaks) / fs
    peak_times = peaks[1:] / fs
    
    # Create a DataFrame of RR intervals
    rr_df = pd.DataFrame({'time': peak_times, 'rr': rr_intervals})
    
    # Define time windows (e.g., every 5 seconds)
    max_time = len(signal) / fs
    bins = np.arange(0, max_time + window_sec, window_sec)
    
    # Group RR intervals by the window they fall into
    rr_df['window'] = pd.cut(rr_df['time'], bins=bins)
    
    # Calculate Coefficient of Variation (CV = std / mean) for each window
    window_stats = rr_df.groupby('window', observed=False)['rr'].agg(['std', 'mean'])
    window_stats['cv'] = window_stats['std'] / window_stats['mean']
    
    # Drop windows with NaN (e.g., 0 or 1 beat in a window)
    window_stats = window_stats.dropna()
    
    if len(window_stats) == 0:
        return 0.0
        
    # Diffusivity is the percentage of windows where CV > threshold
    diffuse_windows = (window_stats['cv'] > cv_threshold).sum()
    total_windows = len(window_stats)
    
    diffusivity_score = diffuse_windows / total_windows
    return diffusivity_score

# 2. Apply the function to all AF recordings
print("Calculating diffusivity for all AF records...")
df_af['diffusivity_score'] = df_af['x'].apply(lambda signal: calculate_diffusivity(signal, fs=fs))

# 3. Plot the Distribution across the dataset
plt.figure(figsize=(10, 6))
plt.hist(df_af['diffusivity_score'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of AF Diffusivity in CPSC 2018 Dataset')
plt.xlabel('Diffusivity Score (0.0 = Highly Sparse, 1.0 = Highly Diffuse)')
plt.ylabel('Number of Recordings')
plt.grid(axis='y', alpha=0.75)

# Replace plt.show() with plt.savefig()
plt.savefig('af_diffusivity_distribution.png', dpi=300, bbox_inches='tight')
print("Plot saved successfully as 'af_diffusivity_distribution.png'")