import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- 1. Load Data ---
try:
    tm = pd.read_csv('tm_noise.csv', header=None).values.flatten()
    ann = pd.read_csv('ann_noise.csv', header=None).values.flatten().astype(int) - 1 
    ecgsig_noise = pd.read_csv('ecgsig_noise.csv', header=None).values.flatten()
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure all required CSV files are present.")
    exit()

Fs = 360 # Hz
level = 5 
wavelet_name = 'sym4'
min_peak_distance_sec = 0.150
min_peak_distance_samples = int(min_peak_distance_sec * Fs)



# =======================================================================
# SECTION 1: MIT-BIH 203 (Noisy Signal Denoising and Detection)
# =======================================================================
ecgsig_203 = ecgsig_noise
ann_203=ann

# ---  Padding for Level 5 SWT ---
level = 5 
min_length = 2**level # 32

N_203_original = len(ecgsig_203)
padding_needed = (min_length - (N_203_original % min_length)) % min_length

# Pad the signal with zeros
ecgsig_203_padded = np.pad(ecgsig_203, (0, padding_needed), 'constant', constant_values=0)
print(f"Original Length: {N_203_original}, Padded Length: {len(ecgsig_203_padded)}") 

# --- PLOT 1: Raw Noisy ECG 203 ---
plt.figure(figsize=(12, 6))
plt.plot(tm[:len(ecgsig_203)], ecgsig_203, label='Noisy ECG Signal')
plt.plot(tm[ann_203], ecgsig_203[ann_203], 'ro', label='Expert Annotations')
plt.xlabel('Seconds')
plt.ylabel('Amplitude')
plt.title('Raw Noisy ECG Signal (MIT-BIH 203)')
plt.legend()
plt.grid(True)
plt.show()

# --- 2. SWT Decomposition and Denoising (Noisy) ---

# Use the padded signal
wt_coeffs_203 = pywt.swt(ecgsig_203_padded, wavelet_name, level=level)

# Create a list of coefficients for reconstruction
denoised_coeffs_203 = []

# Loop through (A, D) tuples, where i=0 is (A5, D5), i=1 is (A4, D4), ..., i=4 is (A1, D1)
for i, (cA, cD) in enumerate(wt_coeffs_203):
    
    # Check if the coefficient is the highest level approximation (A5)
    # i=0 is (A5, D5) tuple. Zero out A5 (baseline wander).
    if i == 0: 
        new_cA = np.zeros_like(cA) # Zero out A5
        new_cD = cD               # Keep D5
        denoised_coeffs_203.append((new_cA, new_cD))
        
    # Check if the coefficient is a high-frequency noise band
    # i=2 is (A3, D3), i=3 is (A2, D2), i=4 is (A1, D1)
    # The detail coefficients (cD) for these upper levels contain the noise.
    elif i >= 2: 
        denoised_coeffs_203.append((cA, np.zeros_like(cD))) # Zero out D3, D2, D1
    
    # This leaves (A4, D4) and (A5, D5) with D5 only.
    else: 
        # Keep A4 and D4 unchanged (or just D4, since we only zeroed A5 above)
        denoised_coeffs_203.append((cA, cD)) 

ecgsig_denoised_203_padded = pywt.iswt(denoised_coeffs_203, wavelet_name)

# --- NEW: TRUNCATE BACK TO ORIGINAL LENGTH ---
ecgsig_denoised_203 = ecgsig_denoised_203_padded[:N_203_original]

# --- 3. R-Peak Indicator and Peak Finding (Noisy) ---
r_peak_indicator_203 = np.abs(ecgsig_denoised_203)**2

# Height tuned for the noisy, denoised signal.
peaks_203_indices, properties_203 = find_peaks(
    r_peak_indicator_203, 
    height=0.4, # INCREASE HEIGHT SLIGHTLY
    distance=min_peak_distance_samples
)

qrspeaks_203 = properties_203['peak_heights']
locs_203 = tm[peaks_203_indices]
ann_203_sq = ann[ann < len(r_peak_indicator_203)] 

# --- PLOT 2: R Waves Localized on Denoised Noisy Signal ---
plt.figure(figsize=(12, 6))
plt.plot(tm[:len(r_peak_indicator_203)], r_peak_indicator_203, label='Denoised Signal Squared')
hwav = plt.plot(locs_203, qrspeaks_203, 'ro', label='Automatic')
hexp = plt.plot(tm[ann_203_sq], r_peak_indicator_203[ann_203_sq], 'k*', label='Expert')
plt.xlabel('Seconds')
plt.title('R-Waves Localized on Denoised Signal (MIT-BIH 203)')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()