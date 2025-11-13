"""
Doppler Processing - frequency shifts due to relative motion
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_doppler_shift(carrier_freq, velocity, wave_speed=3e8):
    """
    calculate doppler frequency shift
    
    carrier_freq: transmitted frequency (Hz)
    velocity: relative velocity in m/s (positive = approaching, negative = receding)
    wave_speed: speed of propagation (default = speed of light)
    
    returns: doppler shift in Hz
    """
    doppler = carrier_freq * velocity / wave_speed
    return doppler


def estimate_velocity_from_doppler(carrier_freq, observed_freq, wave_speed=3e8):
    """
    estimate relative velocity from observed frequency shift
    """
    doppler_shift = observed_freq - carrier_freq
    velocity = doppler_shift * wave_speed / carrier_freq
    return velocity


def apply_doppler_to_signal(signal, doppler_freq, sample_rate):
    """
    apply doppler shift to a complex signal
    """
    t = np.arange(len(signal)) / sample_rate
    # multiply by complex exponential at doppler frequency
    shifted = signal * np.exp(2j * np.pi * doppler_freq * t)
    return shifted


def remove_doppler(signal, doppler_freq, sample_rate):
    """
    compensate for known doppler shift
    """
    t = np.arange(len(signal)) / sample_rate
    # multiply by conjugate of doppler shift
    corrected = signal * np.exp(-2j * np.pi * doppler_freq * t)
    return corrected


# ===== TESTS =====
def test_doppler():
    print("Testing doppler processing...")
    
    # test L1 frequency
    gps_l1 = 1575.42e6  # Hz
    
    # satellite relative velocity
    velocity = 3000  # m/s
    
    doppler = compute_doppler_shift(gps_l1, velocity)

    expected_magnitude = 15e3
    assert 10e3 < abs(doppler) < 20e3, f"doppler seems wrong: {doppler} Hz"
    print(f"✓ Doppler shift for satellite: {doppler/1e3:.2f} kHz (velocity: {velocity} m/s)")
    
    # test velocity estimation
    observed_freq = gps_l1 + 10e3  # 10 kHz shift
    estimated_vel = estimate_velocity_from_doppler(gps_l1, observed_freq)
    
    # should give velocity
    assert 1500 < estimated_vel < 2500, "velocity estimation failed"
    print(f"✓ Velocity estimation: {estimated_vel:.1f} m/s from {10e3} Hz shift")
    
    # test apply and remove doppler
    fs = 10000
    t = np.arange(0, 0.1, 1/fs)
    
    # create IF signal
    if_freq = 1000  # intermediate frequency
    signal = np.exp(2j * np.pi * if_freq * t)
    
    # apply doppler
    doppler_shift = 200  # Hz
    shifted_signal = apply_doppler_to_signal(signal, doppler_shift, fs)
    phase = np.angle(shifted_signal)
    unwrapped = np.unwrap(phase)
    inst_freq = np.diff(unwrapped) * fs / (2*np.pi)
    measured_freq = np.mean(inst_freq)
    
    expected = if_freq + doppler_shift
    error = abs(measured_freq - expected)
    assert error < 5, f"doppler application failed: {measured_freq} vs {expected}"
    print(f"✓ Applied doppler: carrier shifted from {if_freq} to {measured_freq:.1f} Hz")
    
    # test doppler removal
    corrected = remove_doppler(shifted_signal, doppler_shift, fs)
    phase_corr = np.angle(corrected)
    unwrapped_corr = np.unwrap(phase_corr)
    freq_corr = np.diff(unwrapped_corr) * fs / (2*np.pi)
    recovered_freq = np.mean(freq_corr)
    
    error_corr = abs(recovered_freq - if_freq)
    assert error_corr < 5, "doppler removal failed"
    print(f"✓ Removed doppler: frequency corrected to {recovered_freq:.1f} Hz")
    
    # test doppler range for different velocities
    velocities = [-1000, -500, 0, 500, 1000, 2000]  # m/s
    print("\nDoppler shifts for different velocities (at L1 frequency):")
    print("Velocity (m/s) | Doppler (kHz)")
    print("-" * 35)
    for v in velocities:
        d = compute_doppler_shift(gps_l1, v)
        print(f"{v:14.0f} | {d/1e3:13.2f}")
    
    print("\nAll doppler tests passed!\n")
    
    # visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # velocity vs doppler
    vels = np.linspace(-5000, 5000, 100)
    dopplers = [compute_doppler_shift(gps_l1, v) for v in vels]
    
    axes[0,0].plot(vels, np.array(dopplers)/1e3)
    axes[0,0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0,0].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[0,0].set_xlabel('Velocity (m/s)')
    axes[0,0].set_ylabel('Doppler Shift (kHz)')
    axes[0,0].set_title('Doppler vs Velocity (GPS L1)')
    axes[0,0].grid(True)
    
    # signal before and after doppler
    fs = 10000
    t = np.arange(0, 0.05, 1/fs)
    original = np.exp(2j * np.pi * 500 * t)
    shifted = apply_doppler_to_signal(original, 150, fs)
    
    axes[0,1].plot(t[:200], np.real(original[:200]), label='Original (500 Hz)', alpha=0.7)
    axes[0,1].plot(t[:200], np.real(shifted[:200]), label='With Doppler (+150 Hz)')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Real Part')
    axes[0,1].set_title('Signal with Doppler Shift')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # spectrum comparison
    from scipy.fft import fft, fftfreq
    freqs_fft = fftfreq(len(original), 1/fs)
    spec_orig = np.abs(fft(original))
    spec_shifted = np.abs(fft(shifted))
    
    axes[1,0].plot(freqs_fft[:len(freqs_fft)//2], spec_orig[:len(spec_orig)//2], 
                   label='Original', alpha=0.7)
    axes[1,0].plot(freqs_fft[:len(freqs_fft)//2], spec_shifted[:len(spec_shifted)//2], 
                   label='Doppler Shifted')
    axes[1,0].set_xlabel('Frequency (Hz)')
    axes[1,0].set_ylabel('Magnitude')
    axes[1,0].set_title('Frequency Domain Comparison')
    axes[1,0].legend()
    axes[1,0].grid(True)
    axes[1,0].set_xlim([400, 700])
    
    # doppler correction demonstration
    corrected = remove_doppler(shifted, 150, fs)
    
    # phase over time
    phase_shifted = np.unwrap(np.angle(shifted))
    phase_corrected = np.unwrap(np.angle(corrected))
    
    axes[1,1].plot(t[:200], phase_shifted[:200], label='Doppler shifted', alpha=0.7)
    axes[1,1].plot(t[:200], phase_corrected[:200], label='After correction')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Phase (rad)')
    axes[1,1].set_title('Phase Correction')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('test_doppler.png', dpi=100)
    print("Saved test_doppler.png")


if __name__ == '__main__':
    test_doppler()
