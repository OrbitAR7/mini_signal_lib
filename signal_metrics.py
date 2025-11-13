"""
Signal Quality Metrics - measuring signal characteristics
SNR, power, PAPR, and other quality indicators
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_signal_power(signal):
    """
    calculate average signal power
    for complex signals uses magnitude squared
    """
    return np.mean(np.abs(signal)**2)


def compute_snr(signal, noise):
    """
    compute signal to noise ratio
    SNR = 10*log10(signal_power / noise_power)
    """
    sig_power = compute_signal_power(signal)
    noise_power = compute_signal_power(noise)
    
    snr_linear = sig_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db


def estimate_snr_from_noisy_signal(noisy_signal, noise_percentile=10):
    """
    estimate SNR when you only have the noisy signal
    assumes noise floor is at lower percentile of power
    """
    power = np.abs(noisy_signal)**2
    
    # estimate noise floor
    noise_floor = np.percentile(power, noise_percentile)
    signal_power = np.mean(power)
    
    snr_linear = signal_power / noise_floor
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db


def compute_rms(signal):
    """
    root mean square value
    gives "average" magnitude
    """
    return np.sqrt(np.mean(np.abs(signal)**2))


def compute_peak_to_average_ratio(signal):
    """
    PAPR - peak to average power ratio
    important for amplifier design and signal characteristics
    high PAPR means signal has large peaks compared to average
    """
    peak_power = np.max(np.abs(signal)**2)
    avg_power = np.mean(np.abs(signal)**2)
    
    papr = peak_power / avg_power
    papr_db = 10 * np.log10(papr)
    
    return papr_db


def compute_energy(signal):
    """total energy in signal"""
    return np.sum(np.abs(signal)**2)


def compute_crest_factor(signal):
    """
    crest factor = peak amplitude / RMS
    another measure of signal dynamics
    """
    peak = np.max(np.abs(signal))
    rms = compute_rms(signal)
    
    cf = peak / rms
    cf_db = 20 * np.log10(cf)  # use 20*log for amplitude ratio
    
    return cf_db


# ===== TESTS =====
def test_signal_metrics():
    """test signal quality metrics"""
    print("Testing signal quality metrics...")
    
    # test: power calculation
    # sine wave with amplitude A has power A^2/2
    t = np.linspace(0, 1, 1000)
    amplitude = 2.0
    sine = amplitude * np.sin(2*np.pi*10*t)
    
    power = compute_signal_power(sine)
    expected_power = amplitude**2 / 2
    
    error = abs(power - expected_power) / expected_power
    assert error < 0.01, f"power calculation error: {power} vs {expected_power}"
    print(f"✓ Power calculation: {power:.3f} (expected {expected_power:.3f})")
    
    # test: SNR calculation with known signal and noise
    clean_signal = np.sin(2*np.pi*50*t)
    noise = np.random.randn(len(t)) * 0.1
    noisy = clean_signal + noise
    
    snr = compute_snr(clean_signal, noise)
    
    # rough sanity check
    assert 15 < snr < 25, f"SNR seems wrong: {snr} dB"
    print(f"✓ SNR calculation: {snr:.2f} dB")
    
    # test: SNR estimation from noisy signal only
    # create signal with known SNR
    signal_amp = 1.0
    noise_amp = 0.2
    sig = signal_amp * np.sin(2*np.pi*20*t)
    noi = noise_amp * np.random.randn(len(t))
    combined = sig + noi
    
    true_snr = 20 * np.log10(signal_amp / noise_amp)
    estimated_snr = estimate_snr_from_noisy_signal(combined)
    
    error = abs(estimated_snr - true_snr)
    assert error < 5, f"SNR estimation too far off: {estimated_snr} vs {true_snr}"
    print(f"✓ SNR estimation: {estimated_snr:.2f} dB (true: {true_snr:.2f} dB)")
    
    # test: PAPR for different signals
    # sine wave has PAPR = 3 dB
    sine_papr = compute_peak_to_average_ratio(np.sin(2*np.pi*10*t))
    assert 2.8 < sine_papr < 3.2, "sine PAPR should be ~3 dB"
    print(f"✓ Sine wave PAPR: {sine_papr:.2f} dB (expected ~3 dB)")
    
    # constant amplitude has PAPR = 0 dB
    const = np.ones(1000)
    const_papr = compute_peak_to_average_ratio(const)
    assert abs(const_papr) < 0.1, "constant signal PAPR should be 0"
    print(f"✓ Constant signal PAPR: {const_papr:.2f} dB")
    
    # test 5: RMS calculation
    # for sine wave RMS = A/sqrt(2)
    sine_rms = compute_rms(sine)
    expected_rms = amplitude / np.sqrt(2)
    
    error_rms = abs(sine_rms - expected_rms) / expected_rms
    assert error_rms < 0.01, "RMS calculation wrong"
    print(f"✓ RMS: {sine_rms:.3f} (expected {expected_rms:.3f})")
    
    # test 6: energy calculation
    signal_short = np.ones(100)
    energy = compute_energy(signal_short)
    assert energy == 100, "energy should equal number of samples for unit signal"
    print(f"✓ Energy calculation: {energy}")
    
    print("All signal metrics tests passed!\n")
    
    # visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # SNR comparison
    t = np.linspace(0, 1, 1000)
    clean = np.sin(2*np.pi*10*t)
    
    snr_levels = [30, 10, 0]
    for i, snr_target in enumerate(snr_levels):
        # calculate noise level for target SNR
        sig_pow = compute_signal_power(clean)
        noise_pow = sig_pow / (10**(snr_target/10))
        noise = np.random.randn(len(t)) * np.sqrt(noise_pow)
        noisy = clean + noise
        
        if i == 0:
            ax = axes[0, 0]
        elif i == 1:
            ax = axes[0, 1]
        else:
            ax = axes[0, 2]
            
        ax.plot(t[:200], noisy[:200])
        measured_snr = compute_snr(clean, noise)
        ax.set_title(f'SNR = {measured_snr:.1f} dB')
        ax.set_xlabel('Time (s)')
        ax.grid(True)
    
    # PAPR comparison for different signals
    sine_sig = np.sin(2*np.pi*5*t)
    square_sig = np.sign(np.sin(2*np.pi*5*t))
    random_sig = np.random.randn(len(t))
    
    signals = [sine_sig, square_sig, random_sig]
    names = ['Sine', 'Square', 'Random']
    colors = ['blue', 'red', 'green']
    
    for sig, name, color in zip(signals, names, colors):
        papr = compute_peak_to_average_ratio(sig)
        axes[1,0].hist(sig, bins=50, alpha=0.5, label=f'{name} (PAPR={papr:.1f}dB)', 
                       color=color)
    
    axes[1,0].set_title('Amplitude Distributions')
    axes[1,0].set_xlabel('Amplitude')
    axes[1,0].set_ylabel('Count')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # power over time
    window = 100
    moving_power = []
    sig = np.sin(2*np.pi*10*t) + 0.5*np.random.randn(len(t))
    
    for i in range(len(sig) - window):
        power_local = compute_signal_power(sig[i:i+window])
        moving_power.append(power_local)
    
    axes[1,1].plot(moving_power)
    axes[1,1].axhline(np.mean(moving_power), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(moving_power):.3f}')
    axes[1,1].set_title('Instantaneous Power')
    axes[1,1].set_xlabel('Sample')
    axes[1,1].set_ylabel('Power')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # SNR vs different noise levels
    snr_values = []
    noise_levels = np.linspace(0.01, 1, 20)
    
    for noise_level in noise_levels:
        sig = np.sin(2*np.pi*20*t)
        noise = np.random.randn(len(t)) * noise_level
        snr_val = compute_snr(sig, noise)
        snr_values.append(snr_val)
    
    axes[1,2].plot(noise_levels, snr_values, 'o-')
    axes[1,2].set_title('SNR vs Noise Level')
    axes[1,2].set_xlabel('Noise Standard Deviation')
    axes[1,2].set_ylabel('SNR (dB)')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig('test_signal_metrics.png', dpi=100)
    print("Saved test_signal_metrics.png")


if __name__ == '__main__':
    test_signal_metrics()
