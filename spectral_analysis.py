"""
FFT and Spectral Analysis 
FFT, PSD, and frequency estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def compute_fft(signal, sample_rate, n_fft=None):
    """
    compute FFT of signal
    returns frequency bins and complex spectrum
    """
    if n_fft is None:
        n_fft = len(signal)
    
    spectrum = fft(signal, n=n_fft)
    freqs = fftfreq(n_fft, 1/sample_rate)
    
    return freqs, spectrum


def compute_psd(signal, sample_rate, window='hann'):
    """
    power spectral density
    """
    n = len(signal)
    
    # apply window to reduce spectral leakage
    if window == 'hann':
        w = np.hanning(n)
    elif window == 'hamming':
        w = np.hamming(n)
    else:
        w = np.ones(n)
    
    windowed = signal * w
    
    # compute FFT
    freqs, spectrum = compute_fft(windowed, sample_rate)
    
    # PSD
    psd = (np.abs(spectrum)**2) / (sample_rate * n)
    
    return freqs, psd


def find_peak_frequency(signal, sample_rate):
    """find dominant frequency in signal"""
    freqs, spectrum = compute_fft(signal, sample_rate)
    
    # only positive frequencies
    pos_mask = freqs >= 0
    pos_freqs = freqs[pos_mask]
    pos_spec = np.abs(spectrum[pos_mask])
    
    peak_idx = np.argmax(pos_spec)
    return pos_freqs[peak_idx]


def estimate_snr_from_spectrum(freqs, psd, signal_freq, bandwidth=10):
    """
    estimate SNR from PSD
    signal_freq: expected signal frequency
    bandwidth: bandwidth around signal to consider
    """
    # find signal power
    signal_mask = np.abs(freqs - signal_freq) < bandwidth
    signal_power = np.mean(psd[signal_mask])
    
    # noise power (excluding signal region)
    noise_mask = np.abs(freqs - signal_freq) > 2*bandwidth
    noise_power = np.mean(psd[noise_mask & (freqs >= 0)])
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db


# ===== TESTS =====
def test_spectral_analysis():
    """test FFT and spectral analysis"""
    print("Testing spectral analysis...")
    
    # test: single frequency detection
    fs = 1000
    t = np.arange(0, 1, 1/fs)
    freq = 50
    signal = np.sin(2 * np.pi * freq * t)
    
    detected_freq = find_peak_frequency(signal, fs)
    error = abs(detected_freq - freq)
    assert error < 1, f"frequency detection error: {error} Hz"
    print(f"✓ Single frequency detected: {detected_freq:.2f} Hz (actual: {freq} Hz)")
    
    # test: multi-tone signal
    signal2 = (np.sin(2*np.pi*20*t) + 
               0.5*np.sin(2*np.pi*50*t) + 
               0.3*np.sin(2*np.pi*100*t))
    
    freqs, spectrum = compute_fft(signal2, fs)
    pos_freqs = freqs[freqs >= 0]
    pos_spec = np.abs(spectrum[freqs >= 0])
    
    # find peaks
    peaks = []
    for i in range(1, len(pos_spec)-1):
        if pos_spec[i] > pos_spec[i-1] and pos_spec[i] > pos_spec[i+1]:
            if pos_spec[i] > 50:  # threshold
                peaks.append(pos_freqs[i])
    
    expected = [20, 50, 100]
    assert len(peaks) >= 3, "didn't find all tones"
    print(f"✓ Multi-tone detection: found {len(peaks)} peaks")
    
    # test: PSD calculation
    noisy_signal = signal + np.random.randn(len(signal)) * 0.1
    freqs, psd = compute_psd(noisy_signal, fs)
    
    # PSD should be positive
    assert np.all(psd >= 0), "PSD has negative values"
    # should have peak at signal frequency
    peak_freq_idx = np.argmax(psd[freqs >= 0])
    peak_freq = freqs[freqs >= 0][peak_freq_idx]
    assert abs(peak_freq - 50) < 2, "PSD peak at wrong frequency"
    print(f"✓ PSD computation correct, peak at {peak_freq:.1f} Hz")
    
    # test: SNR estimation
    clean = np.sin(2*np.pi*100*t)
    noisy = clean + np.random.randn(len(clean)) * 0.3
    
    freqs, psd = compute_psd(noisy, fs)
    snr_est = estimate_snr_from_spectrum(freqs, psd, 100, bandwidth=5)
    assert snr_est > 0, "SNR estimate seems wrong"
    print(f"✓ SNR estimation: {snr_est:.1f} dB")
    
    print("All spectral analysis tests passed!\n")
    
    # visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # single tone
    fs = 1000
    t = np.arange(0, 1, 1/fs)
    sig1 = np.sin(2*np.pi*50*t)
    freqs, spec = compute_fft(sig1, fs)
    
    axes[0,0].plot(t[:200], sig1[:200])
    axes[0,0].set_title('Time Domain: Single Tone (50 Hz)')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].grid(True)
    
    axes[0,1].plot(freqs[:len(freqs)//2], np.abs(spec[:len(spec)//2]))
    axes[0,1].set_title('Frequency Domain: FFT Magnitude')
    axes[0,1].set_xlabel('Frequency (Hz)')
    axes[0,1].set_ylabel('Magnitude')
    axes[0,1].grid(True)
    
    # multi-tone with noise
    sig2 = (np.sin(2*np.pi*20*t) + 
            0.5*np.sin(2*np.pi*50*t) + 
            0.3*np.sin(2*np.pi*120*t) +
            np.random.randn(len(t))*0.2)
    
    axes[1,0].plot(t[:200], sig2[:200])
    axes[1,0].set_title('Multi-tone Signal + Noise')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].grid(True)
    
    freqs, psd = compute_psd(sig2, fs)
    axes[1,1].semilogy(freqs[:len(freqs)//2], psd[:len(psd)//2])
    axes[1,1].set_title('Power Spectral Density')
    axes[1,1].set_xlabel('Frequency (Hz)')
    axes[1,1].set_ylabel('PSD')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('test_spectral_analysis.png', dpi=100)
    print("Saved test_spectral_analysis.png")


if __name__ == '__main__':
    test_spectral_analysis()
