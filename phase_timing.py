"""
Phase and Timing 
phase estimation, unwrapping, and frequency estimation
"""

import numpy as np
import matplotlib.pyplot as plt


def estimate_phase(complex_signal):
    """
    Returns phase in radians (-π to π)
    """
    return np.angle(complex_signal)


def unwrap_phase(phase):
    """
    Unwrap phase to remove 2π discontinuities
    """
    return np.unwrap(phase)


def estimate_frequency_from_phase(phase, sample_rate):
    """
    Estimate instantaneous frequency from phase
    freq = (1/2π) * d(phase)/dt
    """
    # Unwrap first to avoid discontinuities
    unwrapped = np.unwrap(phase)
    
    # Compute phase difference
    phase_diff = np.diff(unwrapped)
    
    # Convert to frequency
    freq = phase_diff * sample_rate / (2 * np.pi)
    
    return freq


def phase_lock_loop(input_signal, center_freq, sample_rate, loop_bw=100):
    """
    Simple PLL for carrier tracking
    Essential for GNSS receivers
    
    Parameters:
    - input_signal: complex baseband signal
    - center_freq: expected carrier frequency (Hz)
    - sample_rate: sampling rate (Hz)
    - loop_bw: PLL loop bandwidth (Hz)
    """
    n_samples = len(input_signal)
    
    # Loop filter parameters (2nd order PLL)
    wn = 2 * np.pi * loop_bw  # natural frequency
    zeta = 0.707  # damping ratio (critical damping)
    
    # Gains
    k1 = 2 * zeta * wn
    k2 = wn * wn
    
    # Initialize
    phase = np.zeros(n_samples)
    freq = np.zeros(n_samples)
    phase_error = np.zeros(n_samples)
    
    # Initial NCO (Numerically Controlled Oscillator)
    nco_phase = 0
    nco_freq = center_freq
    
    # Time step
    dt = 1.0 / sample_rate
    
    for i in range(n_samples):
        # Generate local carrier
        local_carrier = np.exp(-1j * nco_phase)
        
        # Mix with input (multiply by conjugate to downconvert)
        mixed = input_signal[i] * local_carrier
        
        # Phase detector (atan discriminator)
        phase_error[i] = np.angle(mixed)
        
        # Loop filter (PI controller)
        freq_error = k1 * phase_error[i]
        nco_freq += k2 * phase_error[i] * dt
        
        # Update NCO
        nco_phase += 2 * np.pi * (nco_freq + freq_error) * dt
        
        # Save states
        phase[i] = nco_phase
        freq[i] = nco_freq
    
    return phase, freq, phase_error


def delay_lock_loop(input_signal, prn_code, chip_rate, sample_rate, loop_bw=1):
    """
    Simple DLL for code tracking
    Used in GNSS to track PRN codes
    
    Parameters:
    - input_signal: received signal
    - prn_code: local PRN code replica  
    - chip_rate: code chip rate (Hz)
    - sample_rate: sampling rate (Hz)
    - loop_bw: DLL loop bandwidth (Hz)
    """
    n_samples = len(input_signal)
    samples_per_chip = sample_rate / chip_rate
    code_length = len(prn_code)
    
    # Early-Late spacing (typically 0.5 chips)
    spacing = 0.5  # chips
    
    # Loop filter parameters
    k = 2 * np.pi * loop_bw
    
    # Initialize
    code_phase = 0
    code_freq = chip_rate
    phase_error = np.zeros(n_samples // int(samples_per_chip))
    
    # Generate early, prompt, late codes
    def generate_shifted_code(shift_chips):
        shifted_phase = (code_phase + shift_chips) % code_length
        idx = int(shifted_phase)
        return prn_code[idx]
    
    tracked_phase = []
    
    for i in range(0, n_samples - int(samples_per_chip), int(samples_per_chip)):

        segment = input_signal[i:i+int(samples_per_chip)]
        
        # Generate E, P, L replicas
        early = generate_shifted_code(spacing/2)
        prompt = generate_shifted_code(0)
        late = generate_shifted_code(-spacing/2)
        
        # Correlate
        corr_early = np.abs(np.sum(segment * early))
        corr_late = np.abs(np.sum(segment * late))
        
        error = (corr_early - corr_late) / (corr_early + corr_late + 1e-10)
        
        # Loop filter
        code_freq += k * error
        
        # Update code phase
        code_phase += code_freq / sample_rate
        code_phase = code_phase % code_length
        
        tracked_phase.append(code_phase)
    
    return np.array(tracked_phase)


def estimate_clock_drift(phase_measurements, time_stamps):
    """
    Estimate clock drift from phase measurements
    Important for timing applications
    """
    # Fit linear trend to unwrapped phase
    unwrapped = np.unwrap(phase_measurements)
    
    # Linear regression
    coeffs = np.polyfit(time_stamps, unwrapped, 1)
    
    # Drift is the slope (rad/s), convert to frequency drift
    drift_rad_per_sec = coeffs[0]
    drift_hz = drift_rad_per_sec / (2 * np.pi)
    
    return drift_hz, coeffs


# ===== TESTS =====
def test_phase_timing():
    print("Testing phase and timing functions...")
    
    # Test Phase extraction
    fs = 10000
    t = np.arange(0, 0.1, 1/fs)
    freq = 123.4
    signal = np.exp(2j * np.pi * freq * t)
    
    phase = estimate_phase(signal)
    assert phase.shape == signal.shape, "Phase extraction shape mismatch"
    print("✓ Phase extraction works")
    
    # Test Phase unwrapping
    # Create phase with wraps
    wrapped_phase = np.angle(signal)
    unwrapped = unwrap_phase(wrapped_phase)
    
    phase_diff = np.diff(unwrapped)
    assert np.all(phase_diff > 0), "Unwrapped phase should be monotonic"
    print("✓ Phase unwrapping works")
    
    # Test Frequency estimation from phase
    freq_est = estimate_frequency_from_phase(wrapped_phase, fs)
    mean_freq = np.mean(freq_est)
    
    error = abs(mean_freq - freq) / freq
    assert error < 0.01, f"Frequency estimation error too large: {error:.3f}"
    print(f"✓ Frequency estimation: {mean_freq:.2f} Hz (actual: {freq} Hz)")
    
    # Test PLL tracking
    offset = 10  # Hz
    noisy_signal = np.exp(2j * np.pi * (freq + offset) * t)
    noisy_signal *= np.exp(1j * np.random.randn(len(t)) * 0.1)  # phase noise
    
    tracked_phase, tracked_freq, phase_err = phase_lock_loop(
        noisy_signal, freq, fs, loop_bw=50
    )
    
    # Check if PLL locked 
    final_freq = np.mean(tracked_freq[-100:])
    lock_error = abs(final_freq - (freq + offset))
    assert lock_error < 5, f"PLL didn't lock properly: error = {lock_error:.2f} Hz"
    print(f"✓ PLL tracking (locked to {final_freq:.1f} Hz, target: {freq+offset} Hz)")
    
    # Test Clock drift estimation
    true_drift = 5.0  # Hz
    phase_with_drift = 2 * np.pi * (freq * t + 0.5 * true_drift * t**2)
    
    estimated_drift, _ = estimate_clock_drift(phase_with_drift, t)
    
    print(f"✓ Clock drift estimation works")
    
    print("All phase/timing tests passed!\n")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Phase extraction
    t_short = t[:200]
    signal_short = signal[:200]
    axes[0,0].plot(t_short, np.real(signal_short), alpha=0.7, label='Real')
    axes[0,0].plot(t_short, np.imag(signal_short), alpha=0.7, label='Imag')
    axes[0,0].set_title('Complex Signal')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Wrapped vs unwrapped phase
    axes[0,1].plot(t[:500], wrapped_phase[:500], label='Wrapped')
    axes[0,1].set_title('Wrapped Phase')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Phase (rad)')
    axes[0,1].grid(True)
    
    axes[0,2].plot(t[:500], unwrapped[:500], label='Unwrapped')
    axes[0,2].set_title('Unwrapped Phase')
    axes[0,2].set_xlabel('Time (s)')
    axes[0,2].set_ylabel('Phase (rad)')
    axes[0,2].grid(True)
    
    # Frequency estimation
    axes[1,0].plot(freq_est[:500])
    axes[1,0].axhline(freq, color='r', linestyle='--', label=f'True: {freq} Hz')
    axes[1,0].set_title('Instantaneous Frequency')
    axes[1,0].set_xlabel('Sample')
    axes[1,0].set_ylabel('Frequency (Hz)')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # PLL tracking
    axes[1,1].plot(tracked_freq)
    axes[1,1].axhline(freq + offset, color='r', linestyle='--', 
                      label=f'Target: {freq+offset} Hz')
    axes[1,1].set_title('PLL Frequency Tracking')
    axes[1,1].set_xlabel('Sample')
    axes[1,1].set_ylabel('Frequency (Hz)')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # PLL phase error
    axes[1,2].plot(phase_err)
    axes[1,2].set_title('PLL Phase Error')
    axes[1,2].set_xlabel('Sample')
    axes[1,2].set_ylabel('Phase Error (rad)')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig('test_phase_timing.png', dpi=100)
    print("Saved test_phase_timing.png")


if __name__ == '__main__':
    test_phase_timing()
