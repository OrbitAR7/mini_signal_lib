"""
GPS/GNSS Signal Processing Examples
"""

import numpy as np
import matplotlib.pyplot as plt

# Imports from modules 
from signal_generation import generate_complex_carrier, add_awgn
from correlation import cross_correlation
from signal_metrics import compute_rms, compute_peak_to_average_ratio
from doppler import apply_doppler_to_signal, remove_doppler
from phase_timing import phase_lock_loop

print("="*70)
print("GPS/GNSS SIGNAL PROCESSING EXAMPLES")
print("="*70)

# ===== Example 1: Complete GPS L1 Signal Simulation =====
print("\n1. COMPLETE GPS L1 SIGNAL SIMULATION")
print("-" * 40)

# GPS L1 C/A parameters
L1_FREQ = 1575.42e6       # L1 carrier frequency (Hz)
CHIP_RATE = 1.023e6       # C/A code chip rate (chips/sec)
CODE_LENGTH = 1023        # C/A code length (chips)

# Receiver IF and sampling parameters
if_freq = 4.092e6         # Intermediate frequency after downconversion
fs = 16.368e6            # Sample rate (16x chip rate for good resolution)
duration = 0.001         # 1 ms (one complete C/A code period)

# Generate GPS-like PRN code (Gold code)
np.random.seed(1)  # PRN 1
prn_code = np.random.choice([-1, 1], size=CODE_LENGTH)

# Repeat PRN code to match signal duration
samples_per_chip = fs / CHIP_RATE
total_samples = int(fs * duration)
code_samples = np.repeat(prn_code, int(samples_per_chip))[:total_samples]

# Generate carrier
t, carrier = generate_complex_carrier(if_freq, duration, sample_rate=fs)

# Ensure arrays have same length
min_length = min(len(carrier), len(code_samples))
carrier = carrier[:min_length]
code_samples = code_samples[:min_length]

# Modulate PRN code onto carrier (BPSK modulation)
modulated_signal = carrier * code_samples

# Simulate satellite motion - LEO satellite scenario
satellite_velocity = 7500  # m/s (typical LEO velocity)
doppler_shift = L1_FREQ * satellite_velocity / 3e8  # ~40 kHz for LEO
print(f"LEO Satellite velocity: {satellite_velocity} m/s")
print(f"Doppler shift at L1: {doppler_shift/1e3:.1f} kHz")

# Scale doppler to IF frequency
if_doppler = doppler_shift * (if_freq / L1_FREQ)
signal_with_doppler = apply_doppler_to_signal(modulated_signal, if_doppler, fs)

# Add realistic noise 
cn0_db_hz = 45
bandwidth = 1 / duration  # 1 kHz for 1ms integration
snr_db = cn0_db_hz - 10 * np.log10(bandwidth)
noisy_signal = add_awgn(signal_with_doppler, snr_db)

print(f"C/N0: {cn0_db_hz} dB-Hz")
print(f"SNR (1ms integration): {snr_db:.1f} dB")
print(f"Generated {len(noisy_signal)} samples at {fs/1e6:.3f} MHz")
print(f"Signal: IF={if_freq/1e6:.3f} MHz, Doppler={if_doppler/1e3:.2f} kHz")


# ===== Example 2: GPS Signal Acquisition =====
print("\n2. GPS SIGNAL ACQUISITION")
print("-" * 40)

# Acquisition search space
doppler_range = np.arange(-10e3, 10e3, 500)  # ±10 kHz in 500 Hz steps
code_phases = range(0, CODE_LENGTH)
test_doppler = if_doppler  #

# Remove doppler and IF
corrected = remove_doppler(noisy_signal, test_doppler, fs)
correlation = cross_correlation(corrected[:len(code_samples)], code_samples)

# Find peak
peak_idx = np.argmax(np.abs(correlation))
peak_value = np.abs(correlation[peak_idx])

# Calculate metrics
mean_corr = np.mean(np.abs(correlation))
peak_to_mean = peak_value / mean_corr

print(f"Acquisition Results:")
print(f"  Peak correlation: {peak_value:.3f}")
print(f"  Mean correlation: {mean_corr:.3f}")
print(f"  Peak-to-mean ratio: {peak_to_mean:.1f}")
print(f"  Detection: {'SUCCESS' if peak_to_mean > 3 else 'FAILED'}")


# ===== Example 3: Carrier Phase Tracking (PLL) =====
print("\n3. CARRIER PHASE TRACKING (PLL)")
print("-" * 40)

# Simulate carrier with slow frequency drift
t_pll = np.arange(0, 0.1, 1/fs)  # 100 ms
freq_drift_rate = 100  # Hz/s (frequency rate of change)
instantaneous_freq = if_freq + if_doppler + freq_drift_rate * t_pll
phase = 2 * np.pi * np.cumsum(instantaneous_freq) / fs
carrier_with_drift = np.exp(1j * phase)

# Add phase noise
phase_noise_std = 0.05  # radians
carrier_with_drift *= np.exp(1j * np.random.randn(len(t_pll)) * phase_noise_std)

# Track with PLL
loop_bandwidth = 20  # Hz 
tracked_phase, tracked_freq, phase_error = phase_lock_loop(
    carrier_with_drift, 
    if_freq + if_doppler, 
    fs, 
    loop_bandwidth
)

# Calculate tracking performance
final_freq = np.mean(tracked_freq[-1000:])  
target_freq = if_freq + if_doppler + freq_drift_rate * 0.1  # Freq at end
tracking_error = abs(final_freq - target_freq)

print(f"PLL Configuration:")
print(f"  Loop bandwidth: {loop_bandwidth} Hz")
print(f"  Initial frequency: {(if_freq + if_doppler)/1e6:.4f} MHz")
print(f"  Frequency drift rate: {freq_drift_rate} Hz/s")
print(f"Tracking Results:")
print(f"  Final tracked freq: {final_freq/1e6:.6f} MHz")  
print(f"  Target freq: {target_freq/1e6:.6f} MHz")
print(f"  Tracking error: {tracking_error:.1f} Hz")
print(f"  Phase noise std: {np.std(phase_error[-1000:]):.3f} rad")


# ===== Example 4: Multipath Detection =====
print("\n4. MULTIPATH DETECTION")
print("-" * 40)

# Create direct signal
direct_signal = modulated_signal.copy()
# Create multipath replica 
multipath_delay_chips = 0.6
multipath_delay_samples = multipath_delay_chips / CHIP_RATE * fs
multipath_amplitude = 0.5  # 50% of direct signal
multipath_signal = np.zeros_like(direct_signal)
delay_int = int(np.floor(multipath_delay_samples))
frac = multipath_delay_samples - delay_int
# Simple linear interp for fractional delay
for i in range(delay_int, len(direct_signal)):
    if frac > 0:
        alpha = (i - delay_int) / 1
        alpha = np.clip(alpha, 0, 1)
        multipath_signal[i] = direct_signal[i - delay_int] * multipath_amplitude * (1 - frac + frac * alpha)
    else:
        multipath_signal[i] = direct_signal[i - delay_int] * multipath_amplitude

# Combine direct and multipath
received_with_multipath = direct_signal + multipath_signal

# Downconvert to baseband before correlation (fix)
direct_baseband = remove_doppler(direct_signal, if_freq, fs)
mp_baseband = remove_doppler(received_with_multipath, if_freq, fs)

# Analyze correlation function
len_slice = min(5000, len(code_samples))
corr_direct = cross_correlation(direct_baseband[:len_slice], code_samples[:len_slice])
corr_multipath = cross_correlation(mp_baseband[:len_slice], code_samples[:len_slice])

# Find peaks
main_peak_idx = np.argmax(np.abs(corr_direct))
mp_corr_peak_idx = np.argmax(np.abs(corr_multipath))

# Check for correlation function distortion
correlation_distortion = abs(main_peak_idx - mp_corr_peak_idx)

print(f"Multipath scenario:")
print(f"  Path delay: {multipath_delay_samples/fs*1e9:.1f} ns ({multipath_delay_samples/fs*3e8:.1f} m)")
print(f"  Multipath amplitude: {multipath_amplitude*100:.0f}% of direct")
print(f"  Correlation peak shift: {correlation_distortion} samples")
chip_error = correlation_distortion / samples_per_chip
print(f"  Ranging error: {chip_error * 3e8 / CHIP_RATE:.2f} meters")  # Convert samples to chip fraction


# ===== Example 5: C/N0 Estimation from Real Signal =====
print("\n5. C/N0 ESTIMATION")
print("-" * 40)

# Different integration times for C/N0 estimation
integration_times_ms = [1, 4, 10, 20]

for int_time_ms in integration_times_ms:
    int_duration = int_time_ms * 0.001
    n_samples = int(fs * int_duration)
    signal_segment = noisy_signal[:n_samples]
    # Repeat code for longer integration (coherent, assume no bits)
    repeats = int(int_time_ms)
    code_segment = np.tile(code_samples, repeats)[:n_samples]
    
    # Fix: Power-based SNR for stability
    corr = cross_correlation(signal_segment, code_segment)
    peak = np.max(np.abs(corr))
    # Exclude peak region for noise power
    center = len(corr) // 2
    half_code = len(code_samples) // 2
    side_indices = np.concatenate((np.arange(0, center - half_code), np.arange(center + half_code, len(corr))))
    if len(side_indices) > 0:
        noise_power = np.mean(np.abs(corr[side_indices])**2)
    else:
        noise_power = np.mean(np.abs(corr)**2)
    snr_db = 10 * np.log10(peak**2 / (noise_power + 1e-10))  # Power ratio
    
    # C/N0 = SNR + 10*log10(1/T) for processing gain cancel
    estimated_cn0 = snr_db + 10 * np.log10(1 / int_duration)
    
    print(f"  {int_time_ms:2d} ms integration: C/N0 = {estimated_cn0:.1f} dB-Hz")


# ===== Example 6: Navigation Data Bit Extraction =====
print("\n6. NAVIGATION DATA BIT EXTRACTION")
print("-" * 40)

# GPS nav data is 50 bps, each bit spans 20 C/A code periods (20 ms)
nav_bit_rate = 50  # bits per second
ms_per_bit = 20

# Simulate nav data bits
nav_bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])  # Example preamble-like pattern

# Create 100ms signal with nav data modulation 
code_one_period = code_samples  # 1ms baseband
nav_signal = []
for bit in nav_bits[:5]:  # 5 bits = 100ms
    bit_value = 1 if bit else -1
    for _ in range(20):  # 20 code periods per bit
        nav_signal.extend(code_one_period * bit_value)

nav_signal = np.array(nav_signal[:int(fs * 0.1)])  # Trim to exactly 100ms
extracted_bits = []
samples_per_bit = int(fs * 0.02)  # 20ms

for i in range(5):  # Extract 5 bits
    start = i * samples_per_bit
    end = start + samples_per_bit
    bit_segment = nav_signal[start:end]
    
    # Integrate (sum) over bit period
    integration = np.sum(bit_segment)
    detected_bit = 1 if integration < 0 else 0
    extracted_bits.append(detected_bit)

print(f"Navigation data:")
print(f"  Transmitted bits: {nav_bits[:5]}")
print(f"  Extracted bits:   {np.array(extracted_bits)}")
print(f"  Bit errors: {np.sum(nav_bits[:5] != extracted_bits)}")


# ===== Example 7: Ionospheric Delay Estimation (Dual-Frequency) =====
print("\n7. IONOSPHERIC DELAY ESTIMATION")
print("-" * 40)

# GPS transmits on L1 (1575.42 MHz) and L2 (1227.60 MHz)
L1_FREQ = 1575.42e6
L2_FREQ = 1227.60e6

# Ionospheric delay is inversely proportional to frequency squared
true_range = 20000000  # 20,000 km (typical GPS satellite range)
tec = 50  # Total Electron Content (TECU)

# Ionospheric delay (simplified model)
iono_delay_L1 = 40.3 * tec / (L1_FREQ/1e9)**2  # meters
iono_delay_L2 = 40.3 * tec / (L2_FREQ/1e9)**2  # meters 

# Measured pseudoranges
pr_L1 = true_range + iono_delay_L1
pr_L2 = true_range + iono_delay_L2

# Ionosphere-free combination (fix formula)
alpha = (L1_FREQ / L2_FREQ)**2
pr_iono_free = (alpha * pr_L1 - pr_L2) / (alpha - 1)

print(f"Ionospheric effects:")
print(f"  TEC: {tec} TECU")
print(f"  L1 delay: {iono_delay_L1:.2f} m")
print(f"  L2 delay: {iono_delay_L2:.2f} m")
print(f"  Iono-free range: {pr_iono_free/1e6:.3f} Mm")
print(f"  Correction: {true_range - pr_iono_free:.3f} m residual")


# ===== Example 8: DOP Calculation for Visible Satellites =====
print("\n8. GEOMETRIC DILUTION OF PRECISION (GDOP)")
print("-" * 40)

# Simulate satellite positions (simplified 2D example)
n_satellites = 6
np.random.seed(42)  # For reproducible random
angles = np.linspace(0, 2*np.pi, n_satellites, endpoint=False)

# Satellite unit vectors (direction cosines)
H = np.zeros((n_satellites, 4))  # [x, y, z, clock]
for i in range(n_satellites):
    H[i, 0] = np.cos(angles[i])  # x component
    H[i, 1] = np.sin(angles[i])  # y component
    H[i, 2] = np.random.uniform(0.3, 0.8)  # z component (fix: randomize for better geometry)
    H[i, 3] = 1                   # clock term

# Calculate DOP matrix
try:
    G = np.linalg.inv(H.T @ H)
    GDOP = np.sqrt(np.trace(G))
    PDOP = np.sqrt(G[0,0] + G[1,1] + G[2,2])
    HDOP = np.sqrt(G[0,0] + G[1,1])
    VDOP = np.sqrt(G[2,2])
    TDOP = np.sqrt(G[3,3])
    
    print(f"DOP values for {n_satellites} satellites:")
    print(f"  GDOP: {GDOP:.2f} (Geometric)")
    print(f"  PDOP: {PDOP:.2f} (Position)")
    print(f"  HDOP: {HDOP:.2f} (Horizontal)")
    print(f"  VDOP: {VDOP:.2f} (Vertical)")
    print(f"  TDOP: {TDOP:.2f} (Time)")
except np.linalg.LinAlgError:
    print("  DOP calculation failed (poor geometry)")


# ===== Example 9: Code-Carrier Divergence (Integrity Monitoring) =====
print("\n9. CODE-CARRIER DIVERGENCE MONITORING")
print("-" * 40)

# Simulate code and carrier measurements
time = np.arange(0, 10, 0.1)  # 10 seconds
true_range = 20000000 + 100 * np.sin(2*np.pi*0.1*time)  # Slowly varying

# Code measurements (noisy but unbiased)
code_noise_std = 1.0  # meters
code_measurements = true_range + np.random.randn(len(time)) * code_noise_std

# Carrier measurements (precise but ambiguous)
carrier_noise_std = 0.001  # meters
wavelength = 3e8 / L1_FREQ  
carrier_measurements = true_range + np.random.randn(len(time)) * carrier_noise_std

# Add ionospheric divergence (code delayed, carrier advanced)
iono_variation = 2 * np.sin(2*np.pi*0.05*time)  # Slowly varying iono
code_measurements += iono_variation
carrier_measurements -= iono_variation  # Opposite sign for carrier

# Calculate divergence
divergence = code_measurements - carrier_measurements

# Monitor for cycle slips, Inject a cycle slip
carrier_measurements[50:] += wavelength * 5  # 5 cycle slip
divergence = code_measurements - carrier_measurements  # Recalc after injection
window = np.ones(3) / 3
divergence_smooth = np.convolve(divergence, window, mode='same')

# Detect cycle slip
divergence_diff = np.diff(divergence)
cycle_slip_threshold = wavelength * 2
slip_indices = np.where(np.abs(divergence_diff) > cycle_slip_threshold)[0]
cycle_slip_detected = len(slip_indices) > 0

print(f"Code-Carrier monitoring:")
print(f"  Mean divergence: {np.mean(divergence[:50]):.3f} m")
print(f"  Divergence variation: {np.std(divergence[:50]):.3f} m")
print(f"  Cycle slip detected: {cycle_slip_detected}")
if cycle_slip_detected:
    slip_idx = slip_indices[0] + 1  # Adjust for diff
    slip_magnitude = divergence_diff[slip_indices[0]] / wavelength
    print(f"  Slip at sample {slip_idx}, magnitude: {abs(slip_magnitude):.1f} cycles")  # Abs for magnitude


# ===== Example 10: Signal Quality Monitoring (SQM) =====
print("\n10. SIGNAL QUALITY MONITORING")
print("-" * 40)

# Nominal GPS signal
nominal_signal = modulated_signal[:10000]

# Distorted signal (e.g., due to ionospheric scintillation)
scintillation_amplitude = np.random.lognormal(0, 0.3, 10000)
distorted_signal = nominal_signal * scintillation_amplitude

# Add interference
interference_freq = if_freq + 1e6  # 1 MHz offset
t_short = np.arange(10000) / fs
interference = 0.5 * np.exp(2j * np.pi * interference_freq * t_short)
signal_with_interference = nominal_signal + interference

# Calculate metrics
nominal_papr = compute_peak_to_average_ratio(nominal_signal)
distorted_papr = compute_peak_to_average_ratio(distorted_signal)
interference_papr = compute_peak_to_average_ratio(signal_with_interference)

nominal_rms = compute_rms(nominal_signal)
distorted_rms = compute_rms(distorted_signal)

print(f"Signal Quality Metrics:")
print(f"  Nominal signal:")
print(f"    PAPR: {nominal_papr:.1f} dB")
print(f"    RMS: {nominal_rms:.3f}")
print(f"  With scintillation:")
print(f"    PAPR: {distorted_papr:.1f} dB")
print(f"    RMS: {distorted_rms:.3f}")
print(f"    Degradation: {distorted_papr - nominal_papr:.1f} dB PAPR increase")
print(f"  With interference:")
print(f"    PAPR: {interference_papr:.1f} dB")
print(f"    Alert: {'INTERFERENCE DETECTED' if interference_papr > nominal_papr + 3 else 'Normal'}")