import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 1. Load your local single-lead dataset
df = pd.read_parquet('data/cpsc_2018_labeled_single_lead.parquet')

# Filter for records labeled as Atrial Fibrillation (typically label 2 in CPSC 2018)
af_records = df[df['label'] == 1].index

if len(af_records) == 0:
    print("No AF records found with label=1. Please check your label mapping.")
else:
    # Select a sample AF record to analyze
    sample_idx = af_records[1220]
    signal = np.array(df.loc[sample_idx, 'x'])
    fs = df.loc[sample_idx, 'fs']
    time = np.arange(len(signal)) / fs

    # 2. Extract R-peaks to calculate R-R intervals
    # distance constraints prevent misidentifying T-waves as R-peaks
    std_val = np.std(signal)
    peaks, _ = find_peaks(signal, distance=int(0.3 * fs), prominence=std_val * 0.5)
    
    rr_intervals = np.diff(peaks) / fs * 1000  # Convert to milliseconds
    rr_times = time[peaks[:-1]]

    # 3. Compute Rolling RMSSD (Window size of 5 beats)
    window_size = 5
    rolling_rmssd = []
    rolling_times = []
    for i in range(len(rr_intervals) - window_size + 1):
        window = rr_intervals[i:i+window_size]
        rmssd = np.sqrt(np.mean(np.diff(window) ** 2))
        rolling_rmssd.append(rmssd)
        rolling_times.append(rr_times[i + window_size // 2])

    # 4. Generate Diagnostic Visualization
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Top Plot: ECG Waveform
    axs[0].plot(time, signal, label='ECG Signal', color='black', lw=0.8)
    axs[0].scatter(time[peaks], signal[peaks], color='red', label='Detected R-peaks', zorder=3)
    axs[0].set_title('ECG Signal Track with Detected R-peaks')
    axs[0].set_xlabel('Time (seconds)')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()

    # Middle Plot: Irregularity over time
    axs[1].plot(rolling_times, rolling_rmssd, marker='o', color='blue', linestyle='-')
    axs[1].set_title('Rolling RMSSD (Heart Rate Irregularity Profile Across Timeline)')
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_ylabel('RMSSD (ms)')
    axs[1].grid(True, alpha=0.3)

    # Bottom Plot: Geometry of the rhythm
    axs[2].scatter(rr_intervals[:-1], rr_intervals[1:], color='purple', alpha=0.7)
    axs[2].set_title('Poincaré Plot ($RR_n$ vs $RR_{n+1}$)')
    axs[2].set_xlabel('$RR_n$ (ms)')
    axs[2].set_ylabel('$RR_{n+1}$ (ms)')
    axs[2].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'AF Diffusivity plot for AF sample {sample_idx}')
    plt.savefig(f'plots/pooling/af_diffusivity_proof_{sample_idx}.png', dpi=300)
    print(f"Diagnostic proof plot successfully saved as 'af_diffussivity_proof_{sample_idx}.png'")

    # 5. Global Dataset Statistical Validation
    # Let's check how uniform the rhythm is across ALL AF samples in your dataset
    all_cv_values = []
    for idx in af_records:
        sig = np.array(df.loc[idx, 'x'])
        f_s = df.loc[idx, 'fs']
        pks, _ = find_peaks(sig, distance=int(0.3 * f_s), prominence=np.std(sig) * 0.5)
        if len(pks) < (window_size + 2):
            continue
        rr = np.diff(pks) / f_s * 1000
        rmssd_vals = [np.sqrt(np.mean(np.diff(rr[j:j+window_size]) ** 2)) for j in range(len(rr) - window_size + 1)]
        
        if np.mean(rmssd_vals) > 0:
            # Coefficient of Variation (CV) of irregularity within this single file
            cv = np.std(rmssd_vals) / np.mean(rmssd_vals)
            all_cv_values.append(cv)
            
    print(f"\n--- Dataset Justification Metric ---")
    print(f"Mean Intra-record Irregularity Coefficient of Variation (CV): {np.mean(all_cv_values):.4f}")