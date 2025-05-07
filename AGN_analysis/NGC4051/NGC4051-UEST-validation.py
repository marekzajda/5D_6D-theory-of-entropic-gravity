import numpy as np
from astropy.io import fits
import scipy.signal as signal
import matplotlib.pyplot as plt
import sys
from scipy.special import erf
from scipy.stats import chi2

# Constants
c = 2.99792458e8  # m/s
G = 6.67430e-11  # m^3 kg^-1 s^-2
hbar = 1.0545718e-34  # J s
k_B = 1.380649e-23  # J/K
h = 6.62607015e-34  # J·s
M_sun = 1.989e30  # kg (solar mass)
M_BH = 1.91e6  # Black hole mass [M_sol] for NGC 4051 (přibližná hodnota, upravit podle dat)
M_dot_Edd = (4 * np.pi * G * M_BH * M_sun * 1.673e-27) / (6.652e-29 * c)  # Eddington accretion rate in kg/s
M_dot = 0.12 * M_dot_Edd  # Observed accretion rate (přibližná hodnota)
T_BH = 1.61e-5  # Hawking temperature [K] for NGC 4051 (přibližná hodnota, upravit podle dat)
T_s_astrophysical = 3.02e-5  # s/m, from UEST 5.0 (zůstává stejné pro konzistenci)
r_s = (2 * G * M_BH * M_sun) / (c**2)  # Schwarzschild radius in m
A_BH = 4 * np.pi * r_s**2  # Area of black hole event horizon
A_Planck = (np.sqrt(hbar * G / c**3))**2  # Planck area

# Master variable S_dot (entropy rate)
S_dot_calibrated = 1.11e8  # s, adjusted from cosmic scale Ts to match peaks (přezkoumat pro NGC 4051)
S_dot_calculated = (8.25e94 * k_B) / 0.025  # J/K·s, updated using T_s from UEST 5.0 (přezkoumat)
delta_f_turb = 0.06  # Hz, adjusted turbulence offset
nonlinear_k = 0.9  # Nonlinear factor for harmonics (adjustable)
MAX_SAMPLES = 100000  # Maximum number of samples for FFT to prevent memory issues

def load_data(fits_files):
    all_time, all_rate, all_background, all_fracexp = [], [], [], []
    time_resolution = None
    for file in fits_files:
        try:
            with fits.open(file) as hdul:
                if time_resolution is None:
                    time_resolution = hdul[1].data['TIMEDEL'][0]
                time_data = hdul[1].data['TIME']
                rate_data = sum(hdul[1].data[f'RATE{i}'] for i in range(2, 6))
                back_data = sum(hdul[1].data[f'BACK{i}V'] for i in range(2, 6))
                fracexp_data = hdul[1].data['FRACEXP']
                all_time.append(time_data)
                all_rate.append(rate_data)
                all_background.append(back_data)
                all_fracexp.append(fracexp_data)
        except Exception as e:
            print(f"Warning: Could not load {file} - {str(e)}", file=sys.stderr)
            continue
    if not all_time:
        raise ValueError("No valid data loaded from any files")
    return (np.concatenate(all_time), np.concatenate(all_rate),
            np.concatenate(all_background), np.concatenate(all_fracexp), time_resolution)

def preprocess_data(time, rate, background, fracexp):
    fracexp = np.where(fracexp > 0, fracexp, 1e-10)
    flux = rate / fracexp - background
    valid = np.isfinite(flux) & (flux > 0)
    time, flux = time[valid], flux[valid]
    if len(flux) > MAX_SAMPLES:
        step = len(flux) // MAX_SAMPLES
        time = time[::step][:MAX_SAMPLES]
        flux = flux[::step][:MAX_SAMPLES]
        print(f"Reduced data to {len(flux)} samples to fit memory limit")
    if np.sum(valid) == 0:
        raise ValueError("No finite values in flux array")
    return time, flux

def fit_red_noise(freq, power):
    valid = (freq > 0) & (power > 0)
    if not np.any(valid):
        raise ValueError("No valid positive frequencies for red noise fitting")
    log_freq = np.log10(freq[valid] + 1e-10)
    log_power = np.log10(power[valid] + 1e-10)
    coeffs = np.polyfit(log_freq, log_power, 1)
    alpha = -coeffs[0]
    alpha = max(1.0, min(alpha, 2.0))
    A = 10**coeffs[1]
    red_noise = A * np.power(np.maximum(freq, 1e-10), -alpha)
    return red_noise, A, alpha

def compute_fft(time, flux, time_resolution):
    flux_detrended = signal.detrend(flux, type='linear')
    n = len(flux_detrended)
    window = signal.get_window('hann', n)
    yf = np.fft.fft(flux_detrended * window)
    xf = np.fft.fftfreq(n, d=time_resolution)[:n//2]
    power = 2.0 / n * np.abs(yf[:n//2])
    red_noise, A, alpha = fit_red_noise(xf, power)
    power_corrected = np.where(power > red_noise, power - red_noise, power * 0.1)
    noise_floor = np.median(power_corrected)
    snr = power_corrected / (noise_floor + 1e-10)
    return xf, power_corrected / power_corrected.max(), snr, red_noise, A, alpha

def find_peaks(xf, power):
    median_power = np.median(power)
    std_power = np.std(power[power < 2 * median_power])
    threshold = median_power + 7.2 * std_power  # Increased threshold to reduce number of peaks
    peaks, _ = signal.find_peaks(power, height=threshold, distance=10, prominence=0.05)
    return [(xf[i], power[i]) for i in peaks if xf[i] > 0.01]  # Filtr DC složky (< 0.01 Hz)

def calculate_harmonics(S_dot, c, nyquist_freq, delta_f_turb, T_BH, h, nonlinear_k, n_max=20):
    # Calibrated approach with nonlinear harmonics (S_dot in seconds)
    n_min_cal = int(np.ceil(c / (S_dot * (nyquist_freq - delta_f_turb))))
    n_min_cal = max(1, n_min_cal)
    n_cal = np.arange(n_min_cal, min(n_max + 1, 10000))
    harmonics_cal = (c / (np.power(n_cal, nonlinear_k) * S_dot)) + delta_f_turb
    harmonics_cal = harmonics_cal[harmonics_cal <= nyquist_freq]
    
    # Theoretical approach using calculated S_dot (J/K·s)
    S_dot_calc = S_dot_calculated
    f_base = (S_dot_calc * T_BH) / h  # Base frequency
    n_max_theo = int(np.floor(f_base / (nyquist_freq - delta_f_turb)))
    n_min_theo = 1
    if n_max_theo > 1e9:
        print(f"Warning: n_max_theo capped at 1e9 (was {n_max_theo:.2e}) to prevent memory overflow")
        n_max_theo = int(1e9)
    if n_max_theo < n_min_theo:
        harmonics_theo = np.array([])
    else:
        n_theo = np.arange(n_min_theo, min(n_max_theo + 1, 1000))
        harmonics_theo = (f_base / np.power(n_theo, nonlinear_k)) + delta_f_turb
        harmonics_theo = harmonics_theo[harmonics_theo <= nyquist_freq]
    
    return harmonics_cal, harmonics_theo

def chi2_harmonic(peaks, harmonics_cal, harmonics_theo, sigma=0.08):  # Snížena tolerance na 0.08 Hz
    if len(peaks) == 0 or (len(harmonics_cal) == 0 and len(harmonics_theo) == 0):
        return float('inf'), 0, 0, 1.0, 1.0, float('inf'), 0, 1.0, 1.0
    chi2_val_cal, matches_cal = 0, 0
    chi2_val_theo, matches_theo = 0, 0
    for freq, _ in peaks:
        min_diff_cal = np.min(np.abs(freq - harmonics_cal))
        min_diff_theo = np.min(np.abs(freq - harmonics_theo)) if len(harmonics_theo) > 0 else float('inf')
        if min_diff_cal < sigma:
            chi2_val_cal += (min_diff_cal / sigma)**2
            matches_cal += 1
        if min_diff_theo < sigma:
            chi2_val_theo += (min_diff_theo / sigma)**2
            matches_theo += 1
    p_value_cal = 1.0
    p_value_theo = 1.0
    false_alarm_prob_cal = 1.0
    false_alarm_prob_theo = 1.0
    if matches_cal > 0:
        p_value_cal = 1 - chi2.cdf(chi2_val_cal, df=matches_cal)
        freq_range = 4.167
        trials = len(harmonics_cal)
        prob_per_trial = (2 * sigma) / freq_range
        false_alarm_prob_cal = 1 - (1 - prob_per_trial)**trials
    if matches_theo > 0:
        p_value_theo = 1 - chi2.cdf(chi2_val_theo, df=matches_theo)
        freq_range = 4.167
        trials = len(harmonics_theo)
        prob_per_trial = (2 * sigma) / freq_range
        false_alarm_prob_theo = 1 - (1 - prob_per_trial)**trials
    return (chi2_val_cal, matches_cal, p_value_cal, false_alarm_prob_cal,
            chi2_val_theo, matches_theo, p_value_theo, false_alarm_prob_theo)

def rotation_velocity(r, rho_0=1e7, r_c=10):
    r_safe = np.maximum(r, 1e-10)
    term1 = r
    term2 = r_c * np.sqrt(np.pi) * erf(r / (np.sqrt(2) * r_c))
    arg = (4 * np.pi * G * rho_0 * r_c**2 / r_safe) * (term1 - term2)
    return np.sqrt(np.maximum(arg, 0))

def plot_spectrum(xf, power, peaks, harmonics_cal, harmonics_theo, nyquist_freq, S_dot_cal, S_dot_calc, red_noise, A, alpha, T_BH):
    plt.figure(figsize=(15, 8))
    plt.semilogx(xf, power, 'b-', alpha=0.7, label='Corrected Power Spectrum')
    plt.semilogx(xf, red_noise / red_noise.max(), 'r--', alpha=0.5, label=f'Red Noise (A={A:.2e}, α={alpha:.2f})')
    for h in harmonics_cal[:20]:
        plt.axvline(h, color='gray', linestyle=':', alpha=0.3, label='Calibrated Harmonics' if h == harmonics_cal[0] else "")
    for h in harmonics_theo[:20]:
        plt.axvline(h, color='green', linestyle='--', alpha=0.3, label='Theoretical Harmonics' if h == harmonics_theo[0] else "")
    
    # Select top peaks with emphasis on lower frequencies
    peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
    peaks_below_1hz = [p for p in peaks_sorted if p[0] < 1.0][:5]
    peaks_above_1hz = [p for p in peaks_sorted if p[0] >= 1.0][:15]
    top_peaks = peaks_below_1hz + peaks_above_1hz
    top_peaks = sorted(top_peaks, key=lambda x: x[1], reverse=True)[:20]
    
    for i, (freq, pwr) in enumerate(top_peaks, 1):
        plt.plot(freq, pwr, 'ro', markersize=8)
        plt.text(freq, pwr * 1.1, f'{i}\n{freq:.3f} Hz\n{pwr:.2f}', 
                 ha='center', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.axvline(nyquist_freq, color='g', linestyle='--', label=f'Nyquist: {nyquist_freq:.3f} Hz')
    plt.xlim(0.05, nyquist_freq * 1.2)
    plt.ylim(-0.05, 1.2)
    plt.xlabel('Frequency [Hz]', fontsize=12)
    plt.ylabel('Normalized Power', fontsize=12)
    plt.title(f'NGC 4051 Power Spectrum (S_dot_cal = {S_dot_cal:.2e} s, S_dot_calc = {S_dot_calc:.2e} J/K·s)', fontsize=14)
    plt.grid(True, which="both", alpha=0.5)
    plt.legend(loc='upper right')
    plt.savefig('C:/Users/Marek Zajda/Desktop/NGC4051_Analyza/ngc4051_spectrum_uest.png', dpi=300, bbox_inches='tight')
    plt.close()
    return top_peaks

def main():
    print("="*60)
    print("NGC 4051 UEST 5.0 ANALYSIS".center(60))
    print("="*60)
    try:
        print(f"Using S_dot_calibrated = {S_dot_calibrated:.3e} s and S_dot_calculated = {S_dot_calculated:.3e} J/K·s with delta_f_turb = {delta_f_turb:.3f} Hz")
        fits_files = [
            'C:/Users/Marek Zajda/Desktop/NGC4051_Analyza/P0903580301PNX000SRCTSR8018.FITS',
            'C:/Users/Marek Zajda/Desktop/NGC4051_Analyza/P0903580301PNX000SRCTSR800B.FITS',
            'C:/Users/Marek Zajda/Desktop/NGC4051_Analyza/P0903580301PNX000SRCTSR8007.FITS',
            'C:/Users/Marek Zajda/Desktop/NGC4051_Analyza/P0903580301PNX000SRCTSR8001.FITS'
        ]
        print("\nLoading data files...")
        time, rate, background, fracexp, time_res = load_data(fits_files)
        print(f"Loaded {len(time)} time points with resolution {time_res:.3f} s")
        print("\nPreprocessing data...")
        time, flux = preprocess_data(time, rate, background, fracexp)
        print(f"After filtering: {len(flux)} valid points")
        print("\nPerforming FFT analysis...")
        xf, power, snr, red_noise, A, alpha = compute_fft(time, flux, time_res)
        print(f"Red noise model: A = {A:.2e}, alpha = {alpha:.2f}")
        nyquist = 1 / (2 * time_res)
        print(f"Nyquist frequency: {nyquist:.3f} Hz")
        peaks = find_peaks(xf, power)
        print(f"Found {len(peaks)} significant peaks after red noise subtraction")
        harmonics_cal, harmonics_theo = calculate_harmonics(S_dot_calibrated, c, nyquist, delta_f_turb, T_BH, h, nonlinear_k)
        print(f"Computed {len(harmonics_cal)} calibrated harmonics and {len(harmonics_theo)} theoretical harmonics below Nyquist")
        print("\nFirst 10 Calibrated Harmonics:")
        for i, freq in enumerate(harmonics_cal[:10], 1):
            print(f"f_{i}: {freq:.3f} Hz")
        print("\nTop 20 Peaks:")
        top_peaks = plot_spectrum(xf, power, peaks, harmonics_cal, harmonics_theo, nyquist, S_dot_calibrated, S_dot_calculated, red_noise, A, alpha, T_BH)
        for i, (freq, pwr) in enumerate(top_peaks, 1):
            print(f"{i:2d}. {freq:.3f} Hz (power: {pwr:.2f})")
        chi2_cal, matches_cal, p_value_cal, false_alarm_cal, chi2_theo, matches_theo, p_value_theo, false_alarm_theo = chi2_harmonic(peaks, harmonics_cal, harmonics_theo)
        print(f"\nCalibrated Harmonics: Matches = {matches_cal}, χ² = {chi2_cal:.2f}, P-value = {p_value_cal:.3e}, False Alarm Prob = {false_alarm_cal:.3e}")
        print(f"Theoretical Harmonics: Matches = {matches_theo}, χ² = {chi2_theo:.2f}, P-value = {p_value_theo:.3e}, False Alarm Prob = {false_alarm_theo:.3e}")
        r = np.logspace(0, 2, 100)  # kpc
        v = rotation_velocity(r)
        print("\nGraph saved as 'ngc4051_spectrum_uest.png'")
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        print("Analysis failed. Please check:", file=sys.stderr)
        print("1. Input files exist and are valid FITS", file=sys.stderr)
        print("2. Required columns (TIME, RATE*, BACK*V, FRACEXP) exist", file=sys.stderr)
        print("3. Basic NumPy and SciPy are installed", file=sys.stderr)
        print("4. Available memory is sufficient or reduce MAX_SAMPLES", file=sys.stderr)
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE".center(60))
    print("="*60)

if __name__ == "__main__":
    main()