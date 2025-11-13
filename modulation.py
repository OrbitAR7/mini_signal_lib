"""
Modulation and Demodulation 
AM, FM, BPSK, QPSK modulation schemes
"""

import numpy as np
import matplotlib.pyplot as plt


def amplitude_modulation(carrier_freq, message_signal, sample_rate, mod_index=0.5):
    """
    AM modulation: y(t) = [1 + m(t)] * cos(2πfct)
    """
    t = np.arange(len(message_signal)) / sample_rate
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    
    # modulate
    am_signal = (1 + mod_index * message_signal) * carrier
    
    return am_signal


def am_demodulation(am_signal, carrier_freq, sample_rate):
    """
    simple envelope detector for AM demodulation
    """
    # rectify
    rectified = np.abs(am_signal)
    
    # lowpass filter to get envelope
    from scipy import signal as sp_signal
    nyq = sample_rate / 2
    cutoff = carrier_freq / 5  # cutoff well below carrier
    b, a = sp_signal.butter(4, cutoff/nyq, btype='low')
    envelope = sp_signal.filtfilt(b, a, rectified)
    
    # remove DC component
    demod = envelope - np.mean(envelope)
    
    return demod


def frequency_modulation(carrier_freq, message_signal, sample_rate, freq_deviation=50):
    """
    FM modulation: phase is integral of message
    freq_deviation: maximum frequency deviation in Hz
    """
    t = np.arange(len(message_signal)) / sample_rate
    
    # integrate message to get phase
    phase = 2 * np.pi * freq_deviation * np.cumsum(message_signal) / sample_rate
    
    # modulate
    fm_signal = np.cos(2 * np.pi * carrier_freq * t + phase)
    
    return fm_signal


def bpsk_modulation(bits, carrier_freq, samples_per_bit):
    """
    BPSK: Binary Phase Shift Keying
    0 → phase 0, 1 → phase π
    """
    num_samples = len(bits) * samples_per_bit
    signal = np.zeros(num_samples)
    
    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = start + samples_per_bit
        t = np.arange(samples_per_bit)
        
        # bit 0 = phase 0, bit 1 = phase π
        phase = np.pi * bit
        signal[start:end] = np.cos(2 * np.pi * carrier_freq * t / samples_per_bit + phase)
    
    return signal


def bpsk_demodulation(signal, carrier_freq, samples_per_bit):
    """
    BPSK demodulator using correlation
    """
    num_bits = len(signal) // samples_per_bit
    bits = []
    
    for i in range(num_bits):
        start = i * samples_per_bit
        end = start + samples_per_bit
        segment = signal[start:end]
        
        # correlate with reference signals (0 and π phase)
        t = np.arange(samples_per_bit)
        ref_0 = np.cos(2 * np.pi * carrier_freq * t / samples_per_bit)
        ref_1 = np.cos(2 * np.pi * carrier_freq * t / samples_per_bit + np.pi)
        
        corr_0 = np.sum(segment * ref_0)
        corr_1 = np.sum(segment * ref_1)
        
        # decide based on which correlation is stronger
        bit = 1 if corr_1 > corr_0 else 0
        bits.append(bit)
    
    return np.array(bits)


def qpsk_modulation(symbols, carrier_freq, samples_per_symbol):
    """
    QPSK: Quadrature Phase Shift Keying
    symbols: array of 0,1,2,3 representing 4 phases
    """
    num_samples = len(symbols) * samples_per_symbol
    signal = np.zeros(num_samples, dtype=complex)
    
    # QPSK phase mapping
    phase_map = {0: 0, 1: np.pi/2, 2: np.pi, 3: 3*np.pi/2}
    
    for i, sym in enumerate(symbols):
        start = i * samples_per_symbol
        end = start + samples_per_symbol
        t = np.arange(samples_per_symbol)
        
        phase = phase_map[sym]
        signal[start:end] = np.exp(1j * (2*np.pi*carrier_freq*t/samples_per_symbol + phase))
    
    return signal


# ===== TESTS =====
def test_modulation():
    print("Testing modulation and demodulation...")
    
    # test AM modulation/demodulation
    fs = 8000
    t = np.arange(0, 0.1, 1/fs)
    
    message = np.sin(2*np.pi*10*t)  # 10 Hz message
    
    # modulate
    fc = 1000  # 1 kHz carrier
    am_sig = amplitude_modulation(fc, message, fs, mod_index=0.8)
    
    # demodulate
    demod = am_demodulation(am_sig, fc, fs)
    message_norm = message / np.max(np.abs(message))
    demod_norm = demod / np.max(np.abs(demod))
    
    correlation = np.corrcoef(message_norm, demod_norm)[0, 1]
    assert correlation > 0.8, f"AM demod failed: correlation = {correlation}"
    print(f"✓ AM modulation/demodulation (correlation: {correlation:.3f})")
    
    # test FM modulation
    fm_message = message / np.max(np.abs(message))  # normalize to [-1, 1]
    fm_sig = frequency_modulation(fc, fm_message, fs, freq_deviation=50)
    max_amplitude = np.max(np.abs(fm_sig))
    min_amplitude = np.min(np.abs(fm_sig))
    power = np.mean(fm_sig**2)
    expected_power = 0.5  # power of cos() is 0.5
    assert abs(power - expected_power) < 0.1, f"FM power incorrect: {power:.3f} vs expected {expected_power}"
    print(f"✓ FM modulation (constant envelope: power={power:.3f})")
    
    # test BPSK modulation/demodulation
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1])
    fc_bpsk = 4  # carrier frequency in normalized units
    samps_per_bit = 50
    
    bpsk_sig = bpsk_modulation(bits, fc_bpsk, samps_per_bit)
    
    # add small noise
    noisy_bpsk = bpsk_sig + np.random.randn(len(bpsk_sig)) * 0.1
    
    # demodulate
    demod_bits = bpsk_demodulation(noisy_bpsk, fc_bpsk, samps_per_bit)
    
    # check bit error rate
    errors = np.sum(bits != demod_bits)
    ber = errors / len(bits)
    
    assert ber < 0.1, f"BPSK BER too high: {ber}"
    print(f"✓ BPSK modulation/demodulation (BER: {ber:.3f}, {errors}/{len(bits)} errors)")
    
    # test QPSK modulation
    symbols = np.array([0, 1, 2, 3, 0, 2, 1, 3])
    qpsk_sig = qpsk_modulation(symbols, 4, 50)

    constellation = []
    for i in range(len(symbols)):
        center = i * 50 + 25
        constellation.append(qpsk_sig[center])
    
    constellation = np.array(constellation)

    phases = np.angle(constellation)
    unique_phases = len(np.unique(np.round(phases, 1)))
    assert unique_phases == 4, "QPSK should have 4 phases"
    print(f"✓ QPSK modulation ({unique_phases} distinct phases)")
    
    print("All modulation tests passed!\n")
    
    # visualization
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # AM demonstration
    fs = 8000
    t = np.arange(0, 0.05, 1/fs)
    message = np.sin(2*np.pi*20*t)
    am = amplitude_modulation(1000, message, fs, mod_index=0.8)
    
    axes[0,0].plot(t[:300], message[:300], label='Message')
    axes[0,0].set_title('Message Signal')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    axes[0,1].plot(t[:300], am[:300])
    axes[0,1].set_title('AM Modulated Signal')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].grid(True)
    
    # FM demonstration
    fm = frequency_modulation(1000, message, fs, freq_deviation=200)
    
    axes[1,0].plot(t[:300], fm[:300])
    axes[1,0].set_title('FM Modulated Signal')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].grid(True)
    
    # spectrum
    from scipy.fft import fft, fftfreq
    freqs = fftfreq(len(fm), 1/fs)
    spectrum = np.abs(fft(fm))
    
    axes[1,1].plot(freqs[:len(freqs)//2], spectrum[:len(spectrum)//2])
    axes[1,1].set_title('FM Spectrum (frequency spread)')
    axes[1,1].set_xlabel('Frequency (Hz)')
    axes[1,1].set_xlim([800, 1200])
    axes[1,1].grid(True)
    
    # BPSK
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    bpsk = bpsk_modulation(bits, 4, 50)
    
    axes[2,0].plot(bpsk)
    for i in range(len(bits)+1):
        axes[2,0].axvline(i*50, color='r', linestyle='--', alpha=0.3)
    axes[2,0].set_title(f'BPSK Signal (bits: {bits})')
    axes[2,0].set_xlabel('Sample')
    axes[2,0].grid(True)
    
    # QPSK constellation
    symbols = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    qpsk = qpsk_modulation(symbols, 4, 50)
    
    # sample at centers
    constellation = []
    for i in range(len(symbols)):
        center = i * 50 + 25
        constellation.append(qpsk[center])
    
    const = np.array(constellation)
    
    axes[2,1].scatter(np.real(const), np.imag(const), s=100, c=symbols, cmap='viridis')
    axes[2,1].set_title('QPSK Constellation Diagram')
    axes[2,1].set_xlabel('In-phase (I)')
    axes[2,1].set_ylabel('Quadrature (Q)')
    axes[2,1].axis('equal')
    axes[2,1].grid(True)
    
    # add unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    axes[2,1].add_patch(circle)
    
    plt.tight_layout()
    plt.savefig('test_modulation.png', dpi=100)
    print("Saved test_modulation.png")


if __name__ == '__main__':
    test_modulation()
