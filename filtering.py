"""
Filtering - removing unwanted frequency components
lowpass, highpass, bandpass filters
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def lowpass_filter(input_signal, cutoff_freq, sample_rate, order=4):
    """
    butterworth lowpass filter
    removes high frequency components above cutoff
    """
    nyquist = sample_rate / 2
    normal_cutoff = cutoff_freq / nyquist
    
    b, a = signal.butter(order, normal_cutoff, btype='low')
    filtered = signal.filtfilt(b, a, input_signal)
    
    return filtered


def highpass_filter(input_signal, cutoff_freq, sample_rate, order=4):
    """
    butterworth highpass filter
    removes low frequency components below cutoff
    """
    nyquist = sample_rate / 2
    normal_cutoff = cutoff_freq / nyquist
    
    b, a = signal.butter(order, normal_cutoff, btype='high')
    filtered = signal.filtfilt(b, a, input_signal)
    
    return filtered


def bandpass_filter(input_signal, low_freq, high_freq, sample_rate, order=4):
    """
    butterworth bandpass filter
    keeps only frequencies between low_freq and high_freq
    """
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, input_signal)
    
    return filtered


def moving_average_filter(input_signal, window_size):
    """
    simple moving average
    smooths signal by averaging nearby samples
    """
    kernel = np.ones(window_size) / window_size
    
    # pad to handle edges
    padded = np.pad(input_signal, window_size//2, mode='edge')
    filtered = np.convolve(padded, kernel, mode='valid')
    
    return filtered[:len(input_signal)]


# ===== TESTS =====
def test_filtering():
    """test filter functions"""
    print("Testing filtering...")
    
    fs = 1000
    t = np.arange(0, 1, 1/fs)
    
    # test lowpass removes high frequencies
    low_freq_sig = np.sin(2*np.pi*10*t)
    high_freq_sig = np.sin(2*np.pi*200*t)
    combined = low_freq_sig + high_freq_sig
    
    filtered = lowpass_filter(combined, cutoff_freq=50, sample_rate=fs)
    correlation = np.corrcoef(filtered, low_freq_sig)[0, 1]
    assert correlation > 0.95, "lowpass didn't remove high frequencies"
    print(f"✓ Lowpass filter works (correlation with low freq: {correlation:.3f})")
    
    # test highpass removes low frequencies
    filtered_hp = highpass_filter(combined, cutoff_freq=100, sample_rate=fs)
    correlation_hp = np.corrcoef(filtered_hp, high_freq_sig)[0, 1]
    assert correlation_hp > 0.95, "highpass didn't remove low frequencies"
    print(f"✓ Highpass filter works (correlation with high freq: {correlation_hp:.3f})")
    
    # test bandpass keeps middle frequencies
    low = np.sin(2*np.pi*10*t)
    mid = np.sin(2*np.pi*50*t)
    high = np.sin(2*np.pi*150*t)
    multi = low + mid + high
    
    bp_filtered = bandpass_filter(multi, low_freq=30, high_freq=70, sample_rate=fs)
    corr_mid = np.corrcoef(bp_filtered, mid)[0, 1]
    assert corr_mid > 0.90, "bandpass didn't isolate middle frequency"
    print(f"✓ Bandpass filter works (correlation: {corr_mid:.3f})")
    
    # test moving average smooths noise
    noisy = np.sin(2*np.pi*5*t) + np.random.randn(len(t)) * 0.5
    smoothed = moving_average_filter(noisy, window_size=20)
    
    # smoothed should have less variance than noisy
    assert np.std(smoothed) < np.std(noisy), "moving average didn't smooth"
    print(f"✓ Moving average smooths signal (std: {np.std(noisy):.3f} → {np.std(smoothed):.3f})")
    
    print("All filtering tests passed!\n")
    
    # visualization
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # lowpass example
    t = np.arange(0, 1, 1/1000)
    sig = np.sin(2*np.pi*5*t) + np.sin(2*np.pi*100*t)
    lp = lowpass_filter(sig, cutoff_freq=20, sample_rate=1000)
    
    axes[0,0].plot(t[:300], sig[:300], alpha=0.7, label='Original')
    axes[0,0].plot(t[:300], lp[:300], label='Lowpass (20Hz)')
    axes[0,0].set_title('Lowpass Filter Effect')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # frequency response
    from scipy.fft import fft, fftfreq
    freqs = fftfreq(len(sig), 1/1000)
    axes[0,1].plot(freqs[:len(freqs)//2], np.abs(fft(sig)[:len(freqs)//2]), 
                   alpha=0.7, label='Original')
    axes[0,1].plot(freqs[:len(freqs)//2], np.abs(fft(lp)[:len(freqs)//2]), 
                   label='Filtered')
    axes[0,1].set_title('Frequency Domain')
    axes[0,1].set_xlabel('Frequency (Hz)')
    axes[0,1].set_ylabel('Magnitude')
    axes[0,1].legend()
    axes[0,1].grid(True)
    axes[0,1].set_xlim([0, 150])
    
    # highpass example
    hp = highpass_filter(sig, cutoff_freq=50, sample_rate=1000)
    
    axes[1,0].plot(t[:300], sig[:300], alpha=0.7, label='Original')
    axes[1,0].plot(t[:300], hp[:300], label='Highpass (50Hz)')
    axes[1,0].set_title('Highpass Filter Effect')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    axes[1,1].plot(freqs[:len(freqs)//2], np.abs(fft(sig)[:len(freqs)//2]), 
                   alpha=0.7, label='Original')
    axes[1,1].plot(freqs[:len(freqs)//2], np.abs(fft(hp)[:len(freqs)//2]), 
                   label='Filtered')
    axes[1,1].set_title('Frequency Domain')
    axes[1,1].set_xlabel('Frequency (Hz)')
    axes[1,1].legend()
    axes[1,1].grid(True)
    axes[1,1].set_xlim([0, 150])
    
    # moving average on noisy signal
    noisy = np.sin(2*np.pi*10*t) + np.random.randn(len(t)) * 0.3
    smooth = moving_average_filter(noisy, window_size=20)
    
    axes[2,0].plot(t[:500], noisy[:500], alpha=0.5, label='Noisy')
    axes[2,0].plot(t[:500], smooth[:500], label='Smoothed', linewidth=2)
    axes[2,0].set_title('Moving Average Smoothing')
    axes[2,0].set_xlabel('Time (s)')
    axes[2,0].legend()
    axes[2,0].grid(True)
    
    # bandpass
    multi_sig = (np.sin(2*np.pi*10*t) + 
                 np.sin(2*np.pi*50*t) + 
                 np.sin(2*np.pi*120*t))
    bp = bandpass_filter(multi_sig, 30, 70, 1000)
    
    axes[2,1].plot(freqs[:len(freqs)//2], np.abs(fft(multi_sig)[:len(freqs)//2]), 
                   alpha=0.7, label='Original (3 tones)')
    axes[2,1].plot(freqs[:len(freqs)//2], np.abs(fft(bp)[:len(freqs)//2]), 
                   label='Bandpass (30-70Hz)')
    axes[2,1].set_title('Bandpass Filter')
    axes[2,1].set_xlabel('Frequency (Hz)')
    axes[2,1].legend()
    axes[2,1].grid(True)
    axes[2,1].set_xlim([0, 150])
    
    plt.tight_layout()
    plt.savefig('test_filtering.png', dpi=100)
    print("Saved test_filtering.png")


if __name__ == '__main__':
    test_filtering()
