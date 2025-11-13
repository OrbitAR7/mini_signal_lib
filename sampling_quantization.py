"""
Sampling and Quantization 
Nyquist theorem, aliasing, quantization effects
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp_signal


def resample_signal(signal, old_rate, new_rate):
    """
    resample signal to different sample rate
    uses interpolation
    """
    num_samples = int(len(signal) * new_rate / old_rate)
    resampled = sp_signal.resample(signal, num_samples)
    return resampled


def downsample(signal, factor):
    """
    downsample by keeping every Nth sample
    """
    return signal[::factor]


def quantize_signal(signal, num_bits):
    """
    quantize signal to N bits
    """
    # find signal range
    sig_min = np.min(signal)
    sig_max = np.max(signal)
    
    # number of quantization levels
    num_levels = 2**num_bits
    
    # quantization step
    step = (sig_max - sig_min) / num_levels
    
    # quantize
    quantized = np.round((signal - sig_min) / step) * step + sig_min
    
    return quantized


def compute_quantization_noise(original, quantized):
    """
    compute quantization error/noise
    """
    error = original - quantized
    noise_power = np.mean(error**2)
    return error, noise_power


def check_nyquist_criterion(signal_freq, sample_rate):
    """
    check if sampling rate satisfies Nyquist criterion
    Nyquist rate = 2 * max frequency
    """
    nyquist_rate = 2 * signal_freq
    
    if sample_rate >= nyquist_rate:
        return True, f"OK: {sample_rate} Hz >= {nyquist_rate} Hz (Nyquist rate)"
    else:
        return False, f"ALIASING: {sample_rate} Hz < {nyquist_rate} Hz (Nyquist rate)"


# ===== TESTS =====
def test_sampling_quantization():
    """test sampling and quantization concepts"""
    print("Testing sampling and quantization...")
    
    # test Nyquist criterion
    signal_freq = 100  # Hz
    
    good_rate = 250  # Hz - satisfies Nyquist (>200)
    bad_rate = 150   # Hz - violates Nyquist (<200)
    
    ok, msg = check_nyquist_criterion(signal_freq, good_rate)
    assert ok, "should satisfy Nyquist"
    print(f"✓ Nyquist check: {msg}")
    
    ok2, msg2 = check_nyquist_criterion(signal_freq, bad_rate)
    assert not ok2, "should violate Nyquist"
    print(f"✓ Aliasing detection: {msg2}")
    
    # test demonstrate aliasing
    t_fine = np.linspace(0, 1, 10000)
    freq = 50  # Hz
    signal_analog = np.sin(2*np.pi*freq*t_fine)
    
    # sample at good rate
    fs_good = 200
    t_good = np.arange(0, 1, 1/fs_good)
    sampled_good = np.sin(2*np.pi*freq*t_good)
    
    # sample at bad rate 
    fs_bad = 80  # less than 2*50 = 100 Hz
    t_bad = np.arange(0, 1, 1/fs_bad)
    sampled_bad = np.sin(2*np.pi*freq*t_bad)
    
    print(f"✓ Aliasing demonstrated: {freq}Hz sampled at {fs_bad}Hz")
    
    # test quantization
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2*np.pi*10*t)
    
    # quantize to different bit depths
    quantized_8bit = quantize_signal(signal, 8)
    quantized_4bit = quantize_signal(signal, 4)
    quantized_2bit = quantize_signal(signal, 2)
    
    # more bits = less quantization noise
    _, noise_8 = compute_quantization_noise(signal, quantized_8bit)
    _, noise_4 = compute_quantization_noise(signal, quantized_4bit)
    _, noise_2 = compute_quantization_noise(signal, quantized_2bit)
    
    assert noise_8 < noise_4 < noise_2, "more bits should mean less noise"
    print(f"✓ Quantization noise: 8-bit={noise_8:.6f}, 4-bit={noise_4:.6f}, 2-bit={noise_2:.4f}")
    
    # test resampling
    fs_orig = 1000
    t_orig = np.arange(0, 1, 1/fs_orig)
    sig_orig = np.sin(2*np.pi*20*t_orig)
    
    # resample to lower rate
    fs_new = 500
    resampled = resample_signal(sig_orig, fs_orig, fs_new)
    
    assert len(resampled) == fs_new, "resampled length wrong"
    print(f"✓ Resampling: {len(sig_orig)} samples @ {fs_orig}Hz → {len(resampled)} samples @ {fs_new}Hz")
    
    # test 5: downsampling
    downsampled = downsample(sig_orig, factor=2)
    assert len(downsampled) == len(sig_orig) // 2, "downsampling wrong"
    print(f"✓ Downsampling by 2: {len(sig_orig)} → {len(downsampled)} samples")
    
    print("All sampling/quantization tests passed!\n")
    
    # visualization
    fig = plt.figure(figsize=(14, 10))
    
    # aliasing demonstration
    ax1 = plt.subplot(3, 2, 1)
    t_fine = np.linspace(0, 0.2, 2000)
    freq = 50
    analog = np.sin(2*np.pi*freq*t_fine)
    
    fs_good = 200
    t_good = np.arange(0, 0.2, 1/fs_good)
    sampled_good = np.sin(2*np.pi*freq*t_good)
    
    ax1.plot(t_fine, analog, 'b-', alpha=0.3, label='Analog signal')
    ax1.plot(t_good, sampled_good, 'ro-', markersize=4, label=f'Sampled @ {fs_good}Hz')
    ax1.set_title(f'Good Sampling (fs={fs_good}Hz > 2×{freq}Hz)')
    ax1.set_xlabel('Time (s)')
    ax1.legend()
    ax1.grid(True)
    
    ax2 = plt.subplot(3, 2, 2)
    fs_bad = 70
    t_bad = np.arange(0, 0.2, 1/fs_bad)
    sampled_bad = np.sin(2*np.pi*freq*t_bad)
    
    ax2.plot(t_fine, analog, 'b-', alpha=0.3, label='Analog signal')
    ax2.plot(t_bad, sampled_bad, 'ro-', markersize=4, label=f'Sampled @ {fs_bad}Hz')
    ax2.set_title(f'Aliasing (fs={fs_bad}Hz < 2×{freq}Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.grid(True)
    
    # quantization effects
    t = np.linspace(0, 0.5, 500)
    sig = np.sin(2*np.pi*10*t)
    
    ax3 = plt.subplot(3, 2, 3)
    quant_8 = quantize_signal(sig, 8)
    ax3.plot(t, sig, 'b-', alpha=0.5, label='Original')
    ax3.plot(t, quant_8, 'r-', label='8-bit quantized')
    ax3.set_title('8-bit Quantization')
    ax3.set_xlabel('Time (s)')
    ax3.legend()
    ax3.grid(True)
    
    ax4 = plt.subplot(3, 2, 4)
    quant_3 = quantize_signal(sig, 3)
    ax4.plot(t, sig, 'b-', alpha=0.5, label='Original')
    ax4.plot(t, quant_3, 'r-', label='3-bit quantized')
    ax4.set_title('3-bit Quantization (visible steps)')
    ax4.set_xlabel('Time (s)')
    ax4.legend()
    ax4.grid(True)
    
    # quantization error
    ax5 = plt.subplot(3, 2, 5)
    bits = [2, 3, 4, 5, 6, 7, 8]
    noise_powers = []
    
    for b in bits:
        q = quantize_signal(sig, b)
        _, noise_p = compute_quantization_noise(sig, q)
        noise_powers.append(noise_p)
    
    ax5.semilogy(bits, noise_powers, 'o-')
    ax5.set_title('Quantization Noise vs Bit Depth')
    ax5.set_xlabel('Number of Bits')
    ax5.set_ylabel('Noise Power')
    ax5.grid(True)
    
    # resampling demonstration
    ax6 = plt.subplot(3, 2, 6)
    fs_orig = 1000
    t_orig = np.arange(0, 0.1, 1/fs_orig)
    sig_orig = np.sin(2*np.pi*50*t_orig)
    
    fs_down = 250
    sig_down = resample_signal(sig_orig, fs_orig, fs_down)
    t_down = np.arange(0, 0.1, 1/fs_down)
    
    ax6.plot(t_orig, sig_orig, 'b-', alpha=0.5, label=f'Original ({fs_orig}Hz)')
    ax6.plot(t_down[:len(sig_down)], sig_down, 'ro-', markersize=3, label=f'Resampled ({fs_down}Hz)')
    ax6.set_title('Signal Resampling')
    ax6.set_xlabel('Time (s)')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('test_sampling_quantization.png', dpi=100)
    print("Saved test_sampling_quantization.png")


if __name__ == '__main__':
    test_sampling_quantization()
