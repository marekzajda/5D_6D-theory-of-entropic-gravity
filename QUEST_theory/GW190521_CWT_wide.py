import numpy as np
import matplotlib.pyplot as plt
import os
import wget
from scipy import signal
from pywt import cwt, ContinuousWavelet
import traceback

# Parametry GW190521
fs = 16384  # Vzorkovací frekvence 16 kHz
merger_time = 1242442952.4  # GPS čas sloučení GW190521
start_time = 1242442944  # Start 32sekundového souboru
f_min, f_max = 10, 250  # Frekvenční pásmo pro analýzu (Hz)
time_range = (-2, 4.5)  # Nové časové okno kolem splynutí

# Datové soubory
DATA_FILES = {
    'H1': 'H-H1_GWOSC_16KHZ_R1-1242442952-32.txt',
    'L1': 'L-L1_GWOSC_16KHZ_R1-1242442952-32.txt',
    'V1': 'V-V1_GWOSC_16KHZ_R1-1242442952-32.txt'
}
BASE_URL = "https://www.gw-openscience.org/eventapi/html/GWTC-2.1-confident/GW190521/v4/"

# Načtení dat (jeden sloupec: strain)
def load_gw190521_data(detector, fs=16384):
    filename = DATA_FILES[detector]
    if not os.path.exists(filename):
        print(f"Stahuji {filename}...")
        wget.download(BASE_URL + filename, filename)
    
    try:
        # Načtení jednoho sloupce (strain)
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if not line.startswith('#') and line.strip()]
            strain = np.array([float(line) for line in lines if line.replace('.', '').replace('-', '').replace('e', '').isdigit()])
        print(f"Načtena data pro {detector}: {len(strain)} vzorků")
        # Generování časového sloupce
        time_step = 1 / fs
        time = np.arange(start_time, start_time + len(strain) * time_step, time_step)
        print(f"Generován časový sloupec pro {detector}: {time[0]:.2f} - {time[-1]:.2f} s")
        # Ořezání na nové časové okno kolem merger_time
        idx = (time >= merger_time + time_range[0]) & (time <= merger_time + time_range[1])
        if not np.any(idx):
            print(f"Chyba: Žádná data v časovém okně, ořezávám na střed")
            mid_idx = len(time) // 2
            samples = int((time_range[1] - time_range[0]) * fs)
            idx = slice(max(0, mid_idx - int(-time_range[0] * fs)), 
                       min(len(time), mid_idx + int(time_range[1] * fs)))
            time = time[idx]
            strain = strain[idx]
            print(f"Ořezáno na {len(time)} vzorků, nový rozsah: {time[0]:.2f} - {time[-1]:.2f} s")
        else:
            time = time[idx]
            strain = strain[idx]
        # Bandpass filtr 10–250 Hz
        try:
            b, a = signal.butter(2, [10/(fs/2), 250/(fs/2)], btype='band')
            strain = signal.filtfilt(b, a, strain, method='gust')
            print(f"Bandpass filtr (10–250 Hz) aplikován pro {detector}")
        except Exception as e:
            print(f"Chyba při filtrování pro {detector}: {e}, používám nefilterovaná data")
        # Kontrola variability strainu
        strain = np.nan_to_num(strain, nan=0.0, posinf=0.0, neginf=0.0)
        std_strain = np.std(strain)
        mean_strain = np.mean(strain)
        print(f"Statistiky strainu pro {detector}: std={std_strain:.2e}, mean={mean_strain:.2e}")
        if std_strain < 1e-25:
            print(f"Varování: Nízká variabilita strainu pro {detector} (std={std_strain:.2e}), data mohou být poškozená")
        return time - merger_time, strain  # Vrací čas relativní ke splynutí
    except Exception as e:
        print(f"Chyba při načítání {filename}: {e}")
        with open(filename, 'r') as f:
            print(f"Prvních 5 řádků {filename}:")
            for i, line in enumerate(f):
                if i < 5:
                    print(line.strip())
                else:
                    break
        raise

# Kontinuální vlnková transformace (CWT)
def compute_cwt(time, strain, fs, f_min, f_max, wavelet='mexh'):
    try:
        # Robustní výpočet škál
        scales = np.logspace(np.log10(fs / f_max), np.log10(fs / f_min), 20)  # Škály pro 10–250 Hz
        print(f"Škály pro {wavelet}: min={scales[0]:.2f}, max={scales[-1]:.2f}, počet={len(scales)}")
        wavelet_obj = ContinuousWavelet(wavelet)
        cwtmatr, _ = cwt(strain, scales, wavelet_obj, 1/fs)
        # Manuální mapování škál na frekvence (obcházení scale2frequency)
        central_freq = 1.0  # Přibližná centrální frekvence pro mexh
        freqs = central_freq * fs / scales
        freqs = np.clip(freqs, f_min, f_max)
        print(f"CWT úspěšné pro {wavelet}: scales={len(scales)}, freqs range={freqs[0]:.2f}–{freqs[-1]:.2f} Hz")
        if len(freqs) != cwtmatr.shape[0]:
            print(f"Varování: Nesoulad rozměrů freqs={len(freqs)} a cwtmatr={cwtmatr.shape[0]}, ořezávám")
            min_len = min(len(freqs), cwtmatr.shape[0])
            freqs = freqs[:min_len]
            cwtmatr = cwtmatr[:min_len, :]
        return freqs, cwtmatr
    except Exception as e:
        print(f"Chyba při CWT pro {wavelet}: {str(e)}\n{traceback.format_exc()}")
        return np.array([]), np.array([])

# Vizualizace
def plot_cwt(detector, time, strain, cwt_freqs, cwtmatr, wavelet='mexh'):
    plt.figure(figsize=(12, 8))

    # Časová řada
    plt.subplot(2, 1, 1)
    plt.plot(time, strain, 'b', label=f'{detector} Strain')
    plt.xlabel('Čas (s)')
    plt.ylabel('Strain')
    plt.title(f'GW190521: Časová řada ({detector})')
    plt.legend()

    # CWT spektrogram
    plt.subplot(2, 1, 2)
    if cwtmatr.size > 0:
        plt.imshow(np.abs(cwtmatr)**2, aspect='auto', extent=[time[0], time[-1], cwt_freqs[-1], cwt_freqs[0]],
                   cmap='viridis', interpolation='bilinear')
        plt.colorbar(label='Spektrální výkon')
        plt.ylabel('Frekvence (Hz)')
        plt.xlabel('Čas (s)')
        plt.title(f'CWT Spektrogram ({detector}, {wavelet}, 10–250 Hz)')
    else:
        plt.text(0.5, 0.5, 'CWT selhalo', horizontalalignment='center', verticalalignment='center')
        plt.xlabel('Čas (s)')
        plt.ylabel('Frekvence (Hz)')
        plt.title(f'CWT Spektrogram ({detector}, {wavelet}, 10–250 Hz)')

    plt.tight_layout()
    plt.savefig(f'GW190521_{detector}_CWT_{wavelet}_Analysis.png')
    plt.show()

# Hlavní program
if __name__ == "__main__":
    print(f"Základní CWT analýza gravitačních vln GW190521 (16 kHz), časové okno {time_range[0]}s až {time_range[1]}s")
    
    # Načtení a zpracování dat pro každý detektor
    for detector in ['H1', 'L1', 'V1']:
        print(f"\nZpracovávám detektor {detector}...")
        # Načtení dat
        time, strain = load_gw190521_data(detector, fs)
        if len(strain) == 0:
            print(f"Chyba: Prázdná data pro {detector}, přeskakuji")
            continue
        
        # CWT pro mexh
        cwt_freqs, cwtmatr = compute_cwt(time, strain, fs, f_min, f_max, wavelet='mexh')
        
        # Vizualizace
        if cwtmatr.size > 0:
            plot_cwt(detector, time, strain, cwt_freqs, cwtmatr, wavelet='mexh')
        else:
            print(f"Přeskakuji vizualizaci pro {detector} (mexh) kvůli selhání CWT")