#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Omega Payload Decoder (ASK/FSK/BPSK/QPSK) over GW baseband
----------------------------------------------------------
- Načte H1/L1 strain, předzpracuje (BP + whiten), demoduluje nosnou (955 Hz),
  získá baseband (obálku A(t), fázi φ(t), FM odchylku).
- Základní kanály:
    * AM/ASK: A(t) v okolí 141.7 Hz (Goertzel / matched-filter bank).
    * FSK: dvojice tónů kolem 141.7 Hz (např. fm±Δ), Viterbi dekódování.
    * BPSK/QPSK: Costas loop (carrier + timing recovery), symboly z φ(t).
- Symbol timing recovery: Gardner TED (2x oversampling).
- Fúze H1/L1: vážený průměr podle SNR a koherence.
- Výstupy: dekódované bity, metriky spolehlivosti, jednoduché grafy/CSV.

Použití (příklad):
python omega_payload_decoder.py \
  --h1 H-H1_GWOSC_16KHZ_R1-1126259447-32.txt \
  --l1 L-L1_GWOSC_16KHZ_R1-1126259447-32.txt \
  --fs 16384 --fc 955.0 --fm 141.7 --duration 2.0 \
  --bp 60 1200 --whiten \
  --mode FSK --fsk_deviation 7.0 --symrate 12.5 \
  --outdir OUT_DECODE/GW150914 --event GW150914
"""

import argparse, json, math, os, sys, csv
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import scipy.signal as sg
from scipy.io import wavfile
import matplotlib.pyplot as plt

# ===================== Utility I/O =====================

def load_strain_txt(path: Path):
    s = str(path)
    if s.endswith(".gz"):
        import gzip
        with gzip.open(s, "rt", encoding="utf-8", errors="ignore") as f:
            arr = np.loadtxt(f)
    else:
        arr = np.loadtxt(s)
    if arr.ndim == 1:
        return None, arr.astype(np.float64, copy=False)
    if arr.shape[1] == 1:
        return None, arr[:,0].astype(np.float64, copy=False)
    return arr[:,0].astype(np.float64, copy=False), arr[:,1].astype(np.float64, copy=False)

def center_crop(x: np.ndarray, fs: float, duration_s: float) -> np.ndarray:
    if not duration_s or duration_s <= 0: return x
    need = int(round(duration_s*fs))
    if len(x) <= need: return x
    mid = len(x)//2
    return x[mid-need//2: mid+need//2]

# ===================== Preprocessing =====================

def bandpass(x, fs, lo, hi, order=6):
    sos = sg.butter(order, [max(0.5, lo)/(0.5*fs), hi/(0.5*fs)], btype="band", output="sos")
    return sg.sosfiltfilt(sos, x).astype(np.float64, copy=False)

def whiten_welch(x, fs, nperseg=4096):
    f, Pxx = sg.welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    if len(Pxx) == 0:
        S = np.ones_like(freqs)
    else:
        S = np.interp(freqs, f, Pxx, left=Pxx[0], right=Pxx[-1])
    S = np.maximum(S, np.median(S)*1e-6)
    Y = X/np.sqrt(S + 1e-20)
    y = np.fft.irfft(Y, n=len(x))
    y = (y - np.mean(y)) / (np.std(y) + 1e-9)
    return y

def tukey_norm(x, alpha=0.05):
    w = sg.windows.tukey(len(x), alpha=alpha)
    x = x*w
    m, s = float(np.mean(x)), float(np.std(x) + 1e-9)
    return (x - m)/s

# ===================== Demod (sync I/Q + Hilbert) =====================

def fir_lowpass(cut_hz, fs, taps=4096, beta=8.6):
    from numpy import i0
    nyq = fs/2.0; wc = min(0.999999, float(cut_hz/nyq))
    n = np.arange(taps, dtype=np.float64)
    m = (taps-1)/2.0
    h = np.where(n==m, 2*wc, np.sin(2*np.pi*wc*(n-m))/(np.pi*(n-m)))
    w = i0(beta*np.sqrt(1.0 - ((n-m)/m)**2)) / i0(beta)
    h *= w; h /= np.sum(h); return h

def fftconv(x, h):
    n = len(x)+len(h)-1
    nfft = 1<<(int(n-1).bit_length())
    X = np.fft.rfft(x, nfft); H = np.fft.rfft(h, nfft)
    y = np.fft.irfft(X*H, nfft)
    return y[:len(x)]

def sync_demod(x, fs, fc, lpf_hz=200.0, taps=4096):
    """Synchronous I/Q demodulation -> envelope A(t), phase φ(t)"""
    N = len(x); t = np.arange(N)/fs
    cosc = np.cos(2*np.pi*fc*t); sinc = np.sin(2*np.pi*fc*t)
    I = x * cosc; Q = x * (-sinc)
    h = fir_lowpass(lpf_hz, fs, taps=taps)
    I_lp = fftconv(I, h); Q_lp = fftconv(Q, h)
    A = np.sqrt(np.clip(I_lp*I_lp + Q_lp*Q_lp, 0, None))
    phi = np.unwrap(np.arctan2(Q_lp, I_lp))
    return A, phi

def hilbert_env(x, fs, lpf_hz=200.0):
    a = sg.hilbert(x); A = np.abs(a)
    if lpf_hz and lpf_hz < fs/2:
        taps = fir_lowpass(lpf_hz, fs, taps=2048)
        A = fftconv(A, taps)
    return A

# ===================== PLL / Costas / Gardner =====================

def costas_loop(y, fs, symrate, order="BPSK"):
    """
    Jednoduchý Costasův přijímač:
    - y: baseband complex (zde syntetizujeme z I+jQ po sync_demod → I_lp + j Q_lp)
    - order: 'BPSK' nebo 'QPSK'
    Vrací: symboly, odhad fáze, nosné; potřeba mít více než ~50 symbolů.
    """
    # Normalizace
    y = y / (np.std(y)+1e-9)
    # Parametry smyčky
    N = len(y); sps = max(2, int(round(fs/float(symrate))))
    mu = 0.0  # timing
    out = []
    phase = 0.0
    freq = 0.0
    alpha = 0.01   # loop filter (phase)
    beta  = 0.0005 # loop filter (freq)
    # Gardnerův TED pro timing, jednoduchý
    def gardner_err(xm1, x0, xp1):
        return np.real((np.conj(xm1)+xp1))*np.real(x0) + np.imag((np.conj(xm1)+xp1))*np.imag(x0)
    x = y*np.exp(-1j*phase)
    k = 2
    while k < N-2:
        i = int(mu)
        frac = mu - i
        # jednoduchá lineární interpolace
        x_m1 = x[i-1] if i-1>=0 else x[0]
        x_0  = (1-frac)*x[i] + frac*x[i+1]
        x_p1 = x[i+1]
        # rozhodování
        if order.upper()=="BPSK":
            sym = 1 if np.real(x_0)>=0 else -1
        else:  # QPSK
            ang = np.angle(x_0)
            sym = int(((ang + np.pi) % (2*np.pi)) // (np.pi/2))
        out.append(sym)
        # Costas error (pro BPSK ~ sign(Re)*Im)
        if order.upper()=="BPSK":
            e_c = np.sign(np.real(x_0))*np.imag(x_0)
        else:
            e_c = np.sign(np.real(x_0))*np.imag(x_0) - np.sign(np.imag(x_0))*np.real(x_0)
        # update phase/freq
        freq += beta*e_c
        phase += freq + alpha*e_c
        x = y*np.exp(-1j*phase)
        # timing error (Gardner)
        e_t = gardner_err(x_m1, x_0, x_p1)
        mu += sps + 0.01*e_t
        k = int(mu)
    return np.array(out), phase, freq

# ===================== Goertzel / matched filters =====================

def goertzel_power(x, fs, f0, bw=2.0):
    """Integrovaný výkon v okně okolo f0 pomocí Goertzela (efektivní pro úzké pásmo)."""
    n = len(x); k = int(0.5 + (n*f0)/fs)
    w = (2*np.pi/fs)*k; cosw = np.cos(w); coeff = 2*cosw
    s_prev = 0.0; s_prev2 = 0.0
    for xi in x:
        s = xi + coeff*s_prev - s_prev2
        s_prev2 = s_prev; s_prev = s
    power = s_prev2*s_prev2 + s_prev*s_prev - coeff*s_prev*s_prev2
    return float(power/n)

def make_matched_sine(fs, f0, T):
    t = np.arange(int(round(T*fs)))/fs
    h = np.cos(2*np.pi*f0*t)
    # normalizace na jednotkovou energii
    h = h/np.sqrt(np.sum(h*h)+1e-12)
    return h

def matched_filter_corr(x, h):
    return sg.fftconvolve(x, h[::-1], mode="same")

# ===================== FSK Viterbi (2-tone) =====================

def fsk2_llrs(x, fs, f0a, f0b, symrate):
    """LLR pro 2-FSK: srovnáme korelace na f0a vs f0b v délce 1 symbolu."""
    sps = int(round(fs/symrate))
    hA = make_matched_sine(fs, f0a, T=sps/fs)
    hB = make_matched_sine(fs, f0b, T=sps/fs)
    rA = matched_filter_corr(x, hA); rB = matched_filter_corr(x, hB)
    # vzorkujeme po sps
    L = []
    for k in range(sps//2, len(x)-sps//2, sps):
        LA = np.sum(rA[k-sps//2:k+sps//2]**2)
        LB = np.sum(rB[k-sps//2:k+sps//2]**2)
        L.append(LA - LB)   # >0 → tón A, <0 → tón B
    return np.array(L)

def viterbi_fsk2(llrs, p_err=0.05):
    """Jednoduchý Viterbi pro 2-FSK s penalizací přechodu (HMM se 2 stavy)."""
    N = len(llrs)
    # stavy: 0→toneB, 1→toneA
    # log-likelihoods:
    lA =  +llrs   # pokud bit=1 (A)
    lB =  -llrs   # pokud bit=0 (B)
    # přechody (penalizace změny stavu)
    stay = math.log(1-p_err + 1e-9)
    flip = math.log(p_err + 1e-9)
    dp = np.full((2, N), -1e9); bp = np.zeros((2, N), dtype=np.int8)
    dp[0,0] = lB[0]; dp[1,0] = lA[0]
    for t in range(1,N):
        for s in (0,1):
            # z předchozího 0→s, 1→s
            c0 = dp[0,t-1] + (stay if s==0 else flip)
            c1 = dp[1,t-1] + (stay if s==1 else flip)
            if c0 >= c1:
                dp[s,t] = c0 + (lB[t] if s==0 else lA[t]); bp[s,t]=0
            else:
                dp[s,t] = c1 + (lB[t] if s==0 else lA[t]); bp[s,t]=1
    # backtrack
    s = 1 if dp[1,-1] >= dp[0,-1] else 0
    bits = []
    for t in range(N-1, -1, -1):
        bits.append(s)
        s = bp[s,t]
    bits = bits[::-1]
    return np.array(bits, dtype=int)

# ===================== H1/L1 fusion =====================

def fuse_streams(a, b, w_a, w_b):
    w_a = max(1e-6, float(w_a)); w_b = max(1e-6, float(w_b))
    return (w_a*a + w_b*b) / (w_a + w_b)

# ===================== Main decode =====================

def main():
    ap = argparse.ArgumentParser(description="Omega payload decoder (ASK/FSK/PSK) at fm≈141.7 Hz baseband")
    ap.add_argument("--h1", type=Path, required=True)
    ap.add_argument("--l1", type=Path, required=True)
    ap.add_argument("--fs", type=float, default=16384.0)
    ap.add_argument("--fc", type=float, default=955.0)
    ap.add_argument("--fm", type=float, default=141.7)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--bp", nargs=2, type=float, default=[60.0, 1200.0])
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--lpf", type=float, default=200.0)
    ap.add_argument("--symrate", type=float, default=12.5, help="symbol rate [baud]")
    ap.add_argument("--mode", choices=["ASK","FSK","BPSK","QPSK"], default="FSK")
    ap.add_argument("--fsk_deviation", type=float, default=7.0, help="Δf around fm for 2-FSK (fm±Δ)")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--event", type=str, default="GWXXXXXX")
    args = ap.parse_args()

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)

    # --- Load & preprocess ---
    tH, hH = load_strain_txt(args.h1); tL, hL = load_strain_txt(args.l1)
    hH = center_crop(hH, args.fs, args.duration); hL = center_crop(hL, args.fs, args.duration)
    if args.bp:
        hH = bandpass(hH, args.fs, args.bp[0], args.bp[1])
        hL = bandpass(hL, args.fs, args.bp[0], args.bp[1])
    if args.whiten:
        hH = whiten_welch(hH, args.fs)
        hL = whiten_welch(hL, args.fs)
    hH = tukey_norm(hH, 0.05); hL = tukey_norm(hL, 0.05)

    # --- Sync demod to carrier (955 Hz) ---
    A_H, phi_H = sync_demod(hH, args.fs, args.fc, lpf_hz=args.lpf)
    A_L, phi_L = sync_demod(hL, args.fs, args.fc, lpf_hz=args.lpf)
    # Baseband FM (freq deviation) from phase:
    dphi_H = np.gradient(phi_H) * args.fs/(2*np.pi)
    dphi_L = np.gradient(phi_L) * args.fs/(2*np.pi)

    # --- Weights for fusion (SNR proxy via Welch power at fm) ---
    def bandpower(x, fs, f0, bw=4.0):
        f, Pxx = sg.welch(x, fs=fs, nperseg=min(2048,len(x)))
        mask = (f>=f0-bw/2) & (f<=f0+bw/2)
        return float(np.trapz(Pxx[mask], f[mask])) if np.any(mask) else 0.0
    wH = bandpower(A_H, args.fs, args.fm); wL = bandpower(A_L, args.fs, args.fm)

    # --- Select channel per mode ---
    if args.mode == "ASK":
        # envelope carries data
        xH, xL = A_H, A_L
    elif args.mode == "FSK":
        # use envelope, but we care about tones near fm
        xH, xL = A_H, A_L
    else:  # PSK/QPSK na fázovém kanálu
        # syntetický complex baseband: I+jQ z demodu
        # zde aproximujeme I,Q jako nízkofrek komponenty z demodu
        # (můžeme je vyrobit z I_lp, Q_lp v sync_demod, tady je máme skryté; proto si je zrekonstruujeme)
        # Rekonstrukce: analytic z basebandu A*e^{jφ}
        xH = np.exp(1j*phi_H)
        xL = np.exp(1j*phi_L)

    # --- H1/L1 fusion ---
    if args.mode in ("ASK","FSK"):
        xNET = fuse_streams(xH, xL, wH, wL)
    else:
        # complex fusion se stejnou vahou
        xNET = 0.5*xH + 0.5*xL

    # --- Symbol timing (Gardner) je použito uvnitř Costas pro PSK; u ASK/FSK použijeme jednoduché vzorkování ---
    sps = max(2, int(round(args.fs/args.symrate)))
    # --- Decode ---
    results = {}
    if args.mode == "ASK":
        # matched filter na fm, threshold
        h = make_matched_sine(args.fs, args.fm, T=sps/args.fs)
        r = matched_filter_corr(xNET, h)
        # vzorkování po 1 symbolu
        syms = []
        for k in range(sps//2, len(r)-sps//2, sps):
            val = np.sum(r[k-sps//2:k+sps//2])
            syms.append(1 if val>=0 else 0)
        bits = np.array(syms, dtype=int)
        conf = float(np.mean(np.abs(r))/ (np.std(r)+1e-9))
        results = {"bits": bits.tolist(), "confidence": conf}

    elif args.mode == "FSK":
        fA = args.fm + args.fsk_deviation
        fB = args.fm - args.fsk_deviation
        llr = fsk2_llrs(xNET, args.fs, fA, fB, args.symrate)
        bits = viterbi_fsk2(llr, p_err=0.08)
        # confidence ~ průměrná velikost LLR
        conf = float(np.mean(np.abs(llr)) / (np.std(llr)+1e-9))
        results = {"bits": bits.tolist(), "confidence": conf, "fA": fA, "fB": fB}

    else:  # BPSK/QPSK
        order = args.mode
        # vytvoř komplexní baseband y = A*e^{jφ} ~ zde máme jen e^{jφ}; A lze použít jako váhu
        y = xNET
        bits, sym = [], []
        out, ph, fr = costas_loop(y, args.fs, args.symrate, order=order)
        if order == "BPSK":
            bits = (out>0).astype(int)
        else:
            # QPSK map: 0:(-π,-π/2], 1:(-π/2,0], 2:(0,π/2], 3:(π/2,π]
            # převedeme na 2 bity Gray-kódovaně (jednoduše zde natural)
            # (reálné mapování uprav dle potřeby)
            b0 = (out>>1)&1; b1 = out&1
            bits = np.column_stack([b0,b1]).reshape(-1)
        # confidence: rozptyl úhlové chyby (čím menší, tím lépe)
        conf = float(1.0/(np.var(np.angle(y*np.exp(-1j*ph)))+1e-6))
        results = {"bits": bits.astype(int).tolist(), "confidence": conf}

    # --- Save outputs ---
    with open(outdir/"decoded_bits.txt", "w") as f:
        f.write("".join(str(b) for b in results["bits"]))
    with open(outdir/"decoded.json", "w", encoding="utf-8") as f:
        json.dump({
            "event": args.event,
            "mode": args.mode,
            "symrate": args.symrate,
            "confidence": results["confidence"],
            "bits_count": len(results["bits"]),
            **({"fA": results.get("fA"), "fB": results.get("fB")} if args.mode=="FSK" else {})
        }, f, ensure_ascii=False, indent=2)

    # --- Quick plots ---
    T = np.arange(len(xNET))/args.fs
    plt.figure(figsize=(9,3)); 
    if args.mode in ("ASK","FSK"):
        plt.plot(T, xNET, lw=0.8)
        plt.title(f"{args.event} — Baseband stream (envelope) fused H1/L1")
        plt.ylabel("Amplitude"); plt.xlabel("Time [s]")
    else:
        plt.plot(T, np.unwrap(np.angle(xNET)), lw=0.8)
        plt.title(f"{args.event} — Phase (unwrapped) fused H1/L1")
        plt.ylabel("Phase [rad]"); plt.xlabel("Time [s]")
    plt.tight_layout(); plt.savefig(outdir/"baseband_stream.png", dpi=180); plt.close()

    # spektrum okolo fm
    f, Pxx = sg.welch(xNET if args.mode!="PSK" else np.real(xNET), fs=args.fs, nperseg=min(4096, len(xNET)))
    plt.figure(figsize=(9,3))
    plt.semilogy(f, Pxx)
    plt.axvline(args.fm, ls="--", lw=1)
    plt.xlim(0, max(300, args.lpf))
    plt.title(f"{args.event} — Welch PSD around fm≈{args.fm:.1f} Hz")
    plt.xlabel("Frequency [Hz]"); plt.ylabel("PSD")
    plt.tight_layout(); plt.savefig(outdir/"psd_fm.png", dpi=180); plt.close()

    print(f"[OK] Decoded {len(results['bits'])} bits, confidence={results['confidence']:.3f} → {outdir}")

if __name__ == "__main__":
    main()
