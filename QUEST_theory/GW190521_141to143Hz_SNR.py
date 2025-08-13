import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert

# Parametry GW190521
fs = 16384
merger_time = 1242442952.4
start_time = 1242442944
DATA_FILES = {
    'H1': 'H-H1_GWOSC_16KHZ_R1-1242442952-32.txt',
    'L1': 'L-L1_GWOSC_16KHZ_R1-1242442952-32.txt',
    'V1': 'V-V1_GWOSC_16KHZ_R1-1242442952-32.txt'
}

def load_gw190521_data(detector, fs=16384):
    """Načte a předzpracuje data gravitační vlny pro zvolený detektor."""
    with open(DATA_FILES[detector], 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#') and line.strip()]
        strain = np.array([float(line) for line in lines if line.replace('.', '').replace('-', '').replace('e', '').isdigit()])
    
    time_step = 1 / fs
    time = np.arange(start_time, start_time + len(strain) * time_step, time_step)
    
    # Výběr časového okna kolem mergeru
    idx = (time >= merger_time - 7) & (time <= merger_time + 9)
    time_full = time[idx]
    strain_full = strain[idx]
    
    # Finální okno pro analýzu
    idx_window = (time_full >= merger_time - 2) & (time_full <= merger_time + 4.5)
    time = time_full[idx_window]
    strain = strain_full[idx_window]
    
    # Předzpracování dat
    padding = int(5 * fs)
    strain_padded = np.pad(strain_full, (padding, padding), mode='reflect')
    
    try:
        # Pásmová propust 10-250 Hz
        b, a = signal.butter(2, [10/(fs/2), 250/(fs/2)], btype='band')
        strain_padded = signal.filtfilt(b, a, strain_padded, method='gust')
    except Exception as e:
        print(f"Chyba při filtrování pro {detector}: {e}")
    
    strain = strain_padded[padding:padding + len(strain)]
    strain = np.nan_to_num(strain, nan=0.0, posinf=0.0, neginf=0.0)
    
    return time, strain

def compute_sliding_snr(strain, fs, freq_range=(141, 143), bandwidth=2.0, noise_band=(100, 120), window_size=0.2, step_size=0.1):
    """Vypočte klouzavé SNR pro zadané frekvenční pásmo."""
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)
    
    snr_list = []
    time_points = []
    envelope_list = []
    
    # Průměrování přes frekvenční pásmo
    for target_freq in np.arange(freq_range[0], freq_range[1] + 1, 1):
        narrow_band = [target_freq - bandwidth/2, target_freq + bandwidth/2]
        
        # Filtr pro signál
        b_sig, a_sig = signal.butter(2, [narrow_band[0]/(fs/2), narrow_band[1]/(fs/2)], btype='band')
        signal_filtered = signal.filtfilt(b_sig, a_sig, strain)
        
        # Filtr pro šum
        b_noise, a_noise = signal.butter(2, [noise_band[0]/(fs/2), noise_band[1]/(fs/2)], btype='band')
        noise_filtered = signal.filtfilt(b_noise, a_noise, strain)
        
        # Výpočet obálky
        envelope = np.abs(hilbert(signal_filtered))
        
        for start in range(0, len(strain) - window_samples, step_samples):
            # Výpočet SNR pro aktuální okno
            sig_win = signal_filtered[start:start + window_samples]
            noise_win = noise_filtered[start:start + window_samples]
            env_win = envelope[start:start + window_samples]
            
            signal_power = np.mean(sig_win**2)
            noise_power = np.mean(noise_win**2)
            envelope_power = np.mean(env_win**2)
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = np.nan
            
            # Uložení výsledků
            current_time = (start / fs) - 2  # Čas relativní k mergeru
            
            if len(snr_list) <= (start // step_samples):
                snr_list.append(snr)
                time_points.append(current_time)
                envelope_list.append(envelope_power)
            else:
                # Průměrování přes frekvence
                snr_list[start // step_samples] += snr / (freq_range[1] - freq_range[0] + 1)
                envelope_list[start // step_samples] += envelope_power / (freq_range[1] - freq_range[0] + 1)
    
    return np.array(time_points), np.array(snr_list), np.array(envelope_list)

def plot_results(time_points, snr_db, envelope_power, detector, freq_range):
    """Vykreslí výsledky analýzy s rozsahem osy Y od -30 do +100 dB."""
    plt.figure(figsize=(14, 7))
    
    # Převod na dB a omezení rozsahu
    envelope_db = 10 * np.log10(envelope_power + 1e-60)  # Přidána malá konstanta pro stabilitu
    
    # Vytvoření subplotu s větší vertikální velikostí
    ax = plt.subplot(111)
    
    # Vykreslení křivek s výraznějšími barvami
    line_snr, = ax.plot(time_points, snr_db, 
                       label=f'SNR {freq_range[0]}-{freq_range[1]} Hz (dB)', 
                       linewidth=2.5, color='#1f77b4')
    line_env, = ax.plot(time_points, envelope_db, 
                       label='Envelope Power (dB)', 
                       linewidth=2, linestyle='--', color='#ff7f0e')
    
    # Nastavení osy Y od -30 do 100 dB
    ax.set_ylim(-30, 100)
    ax.set_yticks(np.arange(-30, 101, 10))
    ax.set_xlim(-2, 4.5)
    ax.set_xticks(np.arange(-2, 5, 0.5))
    
    # Vylepšené anotace
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.axvline(0, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label='Merger time')
    
    # Barevné označení fází
    ax.axvspan(-2, 0, color='blue', alpha=0.07, label='Inspiral')
    ax.axvspan(0, 0.2, color='red', alpha=0.07, label='Merger')
    ax.axvspan(0.2, 4.5, color='green', alpha=0.07, label='Ringdown')
    
    # Vylepšený titulek a popisky
    ax.set_title(f'GW190521 Analysis - {detector} Detector\nFrequency Band {freq_range[0]}-{freq_range[1]} Hz', 
                fontsize=14, pad=15, weight='bold')
    ax.set_xlabel('Time relative to merger (s)', fontsize=12, labelpad=10)
    ax.set_ylabel('Power (dB)', fontsize=12, labelpad=10)
    
    # Vylepšený grid
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    
    # Legenda s lepším umístěním a průhledností
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9, 
              bbox_to_anchor=(1, 1), borderaxespad=0.)
    
    # Automatické přizpůsobení layoutu
    plt.tight_layout()
    
    # Uložení s vysokým rozlišením
    plt.savefig(f'GW190521_{detector}_{freq_range[0]}-{freq_range[1]}Hz.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    print("Analýza GW190521 pro frekvenční pásmo 141-143 Hz")
    freq_range = (141, 143)  # Cílové frekvenční pásmo
    
    for detector in ['H1', 'L1', 'V1']:
        print(f"\nZpracovávám data pro detektor {detector}...")
        
        # Načtení a předzpracování dat
        time, strain = load_gw190521_data(detector, fs)
        
        # Výpočet SNR a obálky
        time_points, snr_db, envelope_power = compute_sliding_snr(
            strain, fs,
            freq_range=freq_range,
            bandwidth=2.0,  # Zúžená šířka pásma pro užší frekvenční rozsah
            noise_band=(100, 120),
            window_size=0.2,
            step_size=0.1
        )
        
        # Vykreslení výsledků
        plot_results(time_points, snr_db, envelope_power, detector, freq_range)