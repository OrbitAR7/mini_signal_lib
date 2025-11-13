"""
Signal Generation
basic signal generation (sine, cosine, complex signals)
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_sine(freq, duration, sample_rate=1000, amplitude=1.0, phase=0):
    """generate a sine wave"""
    t = np.arange(0, duration, 1/sample_rate)
    return t, amplitude * np.sin(2 * np.pi * freq * t + phase)


def generate_cosine(freq, duration, sample_rate=1000, amplitude=1.0, phase=0):
    """generate a cosine wave"""
    t = np.arange(0, duration, 1/sample_rate)
    return t, amplitude * np.cos(2 * np.pi * freq * t + phase)


def generate_complex_carrier(freq, duration, sample_rate):
    """
    Generate complex carrier signal.
    
    Parameters:
    - freq: Carrier frequency (Hz)
    - duration: Signal duration (s)
    - sample_rate: Sampling rate (Hz)
    
    Returns:
    - t: Time vector
    - carrier: Complex exponential carrier
    """
    t = np.arange(0, duration, 1 / sample_rate)
    carrier = np.exp(1j * 2 * np.pi * freq * t)
    return t, carrier


def add_awgn(signal, snr_db):
    """
    Add AWGN noise to signal at given SNR (dB).
    
    Parameters:
    - signal: Input signal (complex or real)
    - snr_db: Desired SNR in dB
    
    Returns:
    - noisy_signal: Signal + AWGN
    """
    sig_power = np.mean(np.abs(signal)**2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    sigma = np.sqrt(noise_power / 2)  # For complex noise (real/imag independent)
    noise = sigma * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise


# ===== TESTS =====
def test_signal_generation():
    print("Testing signal generation...")
    
    # test basic sine wave
    t, sine = generate_sine(freq=10, duration=1.0, sample_rate=1000)
    assert len(sine) == 1000, "wrong number of samples"
    assert np.max(sine) <= 1.01, "amplitude too high"
    print("✓ Sine wave generation works")
    
    # test phase shift
    t, sine1 = generate_sine(freq=10, duration=1.0, phase=0)
    t, sine2 = generate_sine(freq=10, duration=1.0, phase=np.pi/2)
    t, cosine = generate_cosine(freq=10, duration=1.0)
    diff = np.mean(np.abs(sine2 - cosine))
    assert diff < 0.01, "phase shift not working"
    print("✓ Phase shift works")
    
    # test complex signal
    t, complex_sig = generate_complex_carrier(freq=50, duration=0.1, sample_rate=1000)
    assert np.iscomplexobj(complex_sig), "should be complex"
    # magnitude should be constant
    mag = np.abs(complex_sig)
    assert np.std(mag) < 0.01, "magnitude should be constant"
    print("✓ Complex carrier generation works")
    
    # test noise addition
    clean = np.ones(1000)
    noisy = add_awgn(clean, snr_db=10)
    # should have added some noise
    assert not np.allclose(clean, noisy), "noise not added"
    print("✓ Noise addition works")
    
    print("All signal generation tests passed!\n")
    
    # make a plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # clean sine
    t, sine = generate_sine(freq=10, duration=0.5, sample_rate=1000)
    axes[0,0].plot(t, sine)
    axes[0,0].set_title('Clean Sine Wave (10 Hz)')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].grid(True)
    
    # noisy sine
    noisy = add_awgn(sine, snr_db=5)
    axes[0,1].plot(t, noisy)
    axes[0,1].set_title('Noisy Sine (SNR=5dB)')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].grid(True)
    
    # complex signal phase
    t, complex_sig = generate_complex_carrier(freq=5, duration=1.0, sample_rate=1000)
    axes[1,0].plot(t, np.real(complex_sig), label='Real')
    axes[1,0].plot(t, np.imag(complex_sig), label='Imag')
    axes[1,0].set_title('Complex Carrier (5 Hz)')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # constellation plot
    axes[1,1].scatter(np.real(complex_sig[::10]), np.imag(complex_sig[::10]), s=1)
    axes[1,1].set_title('Complex Signal Constellation')
    axes[1,1].set_xlabel('Real')
    axes[1,1].set_ylabel('Imag')
    axes[1,1].axis('equal')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('test_signal_generation.png', dpi=100)
    print("Saved test_signal_generation.png")


if __name__ == '__main__':
    test_signal_generation()
