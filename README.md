# Mini Signal Processing Library

just a simple library i made for learning signal processing. each file tests one concept with examples and visualizations.

## What's in here

Each `.py` file has functions for one topic + tests that run when you execute the file:

- **signal_generation.py** - sine waves, complex signals, noise
- **correlation.py** - autocorr, cross-corr, matched filtering
- **spectral_analysis.py** - FFT, PSD, frequency estimation
- **filtering.py** - lowpass, highpass, bandpass filters
- **doppler.py** - doppler shift calculations (satellite stuff)
- **phase_timing.py** - phase unwrap, time delay estimation
- **signal_metrics.py** - SNR, power, PAPR measurements
- **sampling_quantization.py** - Nyquist, aliasing, quantization
- **modulation.py** - AM, FM, BPSK, QPSK

## Notes
- simplified implementations (not optimized)
- doppler calculations assume simple scenarios
- some functions might have edge cases i didn't test
