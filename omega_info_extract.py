#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Omega Information Extractor: demodulation of gravitational-wave carrier to recover baseband info.

Outputs:
- CSV: A_baseband.csv (envelope), phi_baseband.csv (unwrapped phase), fdev_baseband.csv (instantaneous freq deviation)
- WAV: baseband.wav (listen to the modulation)
- PNG: envelope_plot.png, phase_plot.png, psd_baseband.png, coherence_H1L1.png
- JSON: info_summary.json (SNRs, coherence, entropy, MI, band powers)
- Optional: symbols.bin.csv (simple threshold-based "bit" stream for hypothesis testing)

Usage example:
python omega_info_extract.py --h1 H-H1_GWOSC_16KHZ_R1-1126259447-32.txt --l1 L-L1_GWOSC_16KHZ_R1-1126259447-32.txt \
  --fs 16384 --fc 955.0 --fm 141.7 --bp 60 1200 --whiten --duration 2.0 --device cuda \
  --lpf 200 --down 16384->1024 --outdir OUT_INFO/GW150914 --event GW150914
"""

import argparse, json, math, os, sys, csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.signal as sg
from scipy.io import wavfile
import matplotlib.pyplot as plt

import torch
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True

# ---------------- I/O ----------------

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
    if not duration_s or duration_s <= 0:
        return x
    need = int(round(duration_s*fs))
    if len(x) <= need:
        return x
    mid = len(x)//2
    return x[mid-need//2: mid+need//2]

# ---------------- Preproc ----------------

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

# ---------------- Demod (sync I/Q + Hilbert check) ----------------

def fir_lowpass(cut_hz, fs, taps=4096, beta=8.6):
    from numpy import i0
    nyq = fs/2.0
    wc = min(0.999999, float(cut_hz/nyq))
    n = np.arange(taps, dtype=np.float64)
    m = (taps-1)/2.0
    h = np.where(n==m, 2*wc, np.sin(2*np.pi*wc*(n-m))/(np.pi*(n-m)))
    w = i0(beta*np.sqrt(1.0 - ((n-m)/m)**2)) / i0(beta)
    h *= w
    h /= np.sum(h)
    return h

def fftconv(x, h):
    n = len(x)+len(h)-1
    nfft = 1<<(int(n-1).bit_length())
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)
    y = np.fft.irfft(X*H, nfft)
    return y[:len(x)]

def sync_demod(env_fs, x, fs, fc, lpf_hz=200.0, taps=4096, use_cuda=False):
    """Synchronous I/Q demodulation -> envelope A(t), phase φ(t)"""
    N = len(x)
    t = np.arange(N)/fs
    cosc = np.cos(2*np.pi*fc*t)
    sinc = np.sin(2*np.pi*fc*t)
    I = x * cosc
    Q = x * (-sinc)
    h = fir_lowpass(lpf_hz, fs, taps=taps)
    I_lp = fftconv(I, h)
    Q_lp = fftconv(Q, h)
    A = np.sqrt(np.clip(I_lp*I_lp + Q_lp*Q_lp, 0, None))
    phi = np.unwrap(np.arctan2(Q_lp, I_lp))
    # resample to env_fs
    if env_fs and env_fs < fs:
        g = int(round(fs/env_fs))
        if g>1:
            A = sg.decimate(A, g, ftype="fir", zero_phase=True)
            phi = sg.decimate(phi, g, ftype="fir", zero_phase=True)
            out_fs = fs/g
        else:
            out_fs = fs
    else:
        out_fs = fs
    return A, phi, out_fs

def hilbert_env(x, fs, env_fs=None, lpf_hz=200.0):
    """Analytic-signal envelope as a cross-check."""
    analytic = sg.hilbert(x)
    A = np.abs(analytic)
    # optional low-pass on envelope
    if lpf_hz and lpf_hz < fs/2:
        taps = fir_lowpass(lpf_hz, fs, taps=2048)
        A = fftconv(A, taps)
    if env_fs and env_fs < fs:
        g = int(round(fs/env_fs))
        if g>1:
            A = sg.decimate(A, g, ftype="fir", zero_phase=True)
            fs = fs/g
    return A, fs

# ---------------- Info metrics ----------------

def bandpower(x, fs, lo, hi):
    f, Pxx = sg.welch(x, fs=fs, nperseg=min(4096, len(x)))
    mask = (f>=lo) & (f<=hi)
    return float(np.trapezoid(Pxx[mask], f[mask])) if np.any(mask) else 0.0

def baseband_snr_at(A, fs, f0, bw=4.0):
    sig = bandpower(A, fs, f0-bw/2, f0+bw/2)
    guard = bandpower(A, fs, max(0.5, f0-10.0), f0-6.0) + bandpower(A, fs, f0+6.0, f0+10.0)
    return float(sig/((guard/2)+1e-12)), sig

def coherence_xy(x, y, fs, nperseg=1024):
    f, Cxy = sg.coherence(x, y, fs=fs, nperseg=min(nperseg, len(x)))
    return f, Cxy

def entropy_rate(x):
    # crude: normalized spectral entropy
    f, Pxx = sg.welch(x, nperseg=min(4096,len(x)))
    P = Pxx/np.sum(Pxx + 1e-20)
    H = -np.sum(P*np.log(P+1e-20))
    Hmax = np.log(len(P))
    return float(H/Hmax)

def mutual_information_binary(a, b, thr_a=None, thr_b=None):
    # quick MI on binarized streams (for H1/L1 envelope)
    thr_a = thr_a if thr_a is not None else np.median(a)
    thr_b = thr_b if thr_b is not None else np.median(b)
    A = (a>thr_a).astype(int); B=(b>thr_b).astype(int)
    pa = np.mean(A); pb=np.mean(B)
    pab = np.mean((A==1)&(B==1))
    p00 = np.mean((A==0)&(B==0))
    p01 = np.mean((A==0)&(B==1))
    p10 = np.mean((A==1)&(B==0))
    def h(p): return -p*np.log2(p+1e-20)
    H = h(pab)+h(p01)+h(p10)+h(p00)
    Ha = h(pa)+h(1-pa); Hb=h(pb)+h(1-pb)
    return float(Ha+Hb-H)

def simple_symbol_decode(A, fs, bit_rate=10.0):
    # toy: envelope thresholding with integrate-and-dump at given bit_rate
    N = len(A); Tb = int(round(fs/bit_rate))
    if Tb<=1: Tb=2
    # normalize
    A = (A - np.median(A))/ (np.std(A)+1e-9)
    bits=[]
    for k in range(0, N, Tb):
        seg = A[k:k+Tb]
        if len(seg)<Tb: break
        bits.append(1 if np.mean(seg)>=0 else 0)
    return np.array(bits, dtype=int)

# ---------------- Main pipeline ----------------

def main():
    ap = argparse.ArgumentParser(description="Extract baseband info (A(t), phi(t), fdev) from GW carrier.")
    ap.add_argument("--h1", type=Path, required=True)
    ap.add_argument("--l1", type=Path, required=True)
    ap.add_argument("--v1", type=Path, default=None)
    ap.add_argument("--fs", type=float, default=16384.0)
    ap.add_argument("--fc", type=float, default=955.0)
    ap.add_argument("--fm", type=float, default=141.7)
    ap.add_argument("--bp", nargs=2, type=float, default=[60.0, 1200.0],
                    help="Band-pass on raw strain before demod (default 60–1200 Hz)")
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--lpf", type=float, default=200.0, help="LPF cutoff for baseband (Hz)")
    ap.add_argument("--down", type=str, default="16384->1024", help="Downsample raw to baseband fs (e.g., 16384->1024)")
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--event", type=str, default="GWXXXXXX")
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    # parse down
    if "->" in args.down:
        fs_in, fs_env = [float(x) for x in args.down.split("->")]
    else:
        fs_in, fs_env = args.fs, min(2048.0, args.fs)

    # load
    tH, hH = load_strain_txt(args.h1)
    tL, hL = load_strain_txt(args.l1)

    # crop to on-source
    hH = center_crop(hH, args.fs, args.duration)
    hL = center_crop(hL, args.fs, args.duration)

    # preproc
    if args.bp:
        hH = bandpass(hH, args.fs, args.bp[0], args.bp[1])
        hL = bandpass(hL, args.fs, args.bp[0], args.bp[1])
    if args.whiten:
        hH = whiten_welch(hH, args.fs)
        hL = whiten_welch(hL, args.fs)

    # window & normalize
    def prep(x):
        x = tukey_norm(x, 0.05)
        m, s = float(np.mean(x)), float(np.std(x)+1e-9)
        return (x-m)/s
    hH = prep(hH); hL = prep(hL)

    # synchronous demod (I/Q)
    A_H, phi_H, fs_env_H = sync_demod(fs_env, hH, args.fs, args.fc, lpf_hz=args.lpf)
    A_L, phi_L, fs_env_L = sync_demod(fs_env, hL, args.fs, args.fc, lpf_hz=args.lpf)

    assert int(round(fs_env_H)) == int(round(fs_env_L))
    fsb = fs_env_H

    # instantaneous frequency deviation (FM) from phase
    dphi_H = np.gradient(phi_H) * fsb/(2*np.pi)
    dphi_L = np.gradient(phi_L) * fsb/(2*np.pi)

    # matched filter on fm (optional metric)
    snrA_H, pwA_H = baseband_snr_at(A_H, fsb, args.fm, bw=4.0)
    snrA_L, pwA_L = baseband_snr_at(A_L, fsb, args.fm, bw=4.0)

    # H1-L1 coherence on baseband
    fC, Cxy = sg.coherence(A_H, A_L, fsb, nperseg=min(1024, len(A_H)))
    coh_around_fm = float(np.nanmean(Cxy[(fC>=args.fm-2) & (fC<=args.fm+2)]) if np.any((fC>=args.fm-2)&(fC<=args.fm+2)) else 0.0)

    # entropy + MI
    ent_H = entropy_rate(A_H)
    ent_L = entropy_rate(A_L)
    MI = mutual_information_binary(A_H, A_L)

    # save CSVs
    t = np.arange(len(A_H))/fsb
    np.savetxt(outdir/"A_baseband.csv", np.c_[t, A_H, A_L], delimiter=",", header="t,A_H,A_L", comments="")
    np.savetxt(outdir/"phi_baseband.csv", np.c_[t, phi_H, phi_L], delimiter=",", header="t,phi_H,phi_L", comments="")
    np.savetxt(outdir/"fdev_baseband.csv", np.c_[t, dphi_H, dphi_L], delimiter=",", header="t,fdev_H,fdev_L", comments="")

    # write WAV (normalize to 0.9)
    Af = (A_H - np.mean(A_H))/ (np.max(np.abs(A_H))+1e-12)
    wav = np.int16(np.clip(0.9*Af, -1, 1)*32767)
    wavfile.write(outdir/"baseband.wav", int(round(fsb)), wav)

    # quick symbol decode (toy)
    bits = simple_symbol_decode(A_H, fsb, bit_rate=10.0)
    with open(outdir/"symbols.bin.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["bit_index","bit"])
        for i,b in enumerate(bits): w.writerow([i,int(b)])

    # plots
    plt.figure(figsize=(9,3)); plt.plot(t, A_H, label="A_H"); plt.plot(t, A_L, alpha=0.6, label="A_L")
    plt.xlabel("Time [s]"); plt.ylabel("Envelope A(t)"); plt.title(f"{args.event} — Baseband Envelope")
    plt.legend(); plt.tight_layout(); plt.savefig(outdir/"envelope_plot.png", dpi=180); plt.close()

    plt.figure(figsize=(9,3)); plt.plot(t, phi_H, label="phi_H"); plt.plot(t, phi_L, alpha=0.6, label="phi_L")
    plt.xlabel("Time [s]"); plt.ylabel("Phase φ(t) [rad]"); plt.title(f"{args.event} — Unwrapped Phase")
    plt.legend(); plt.tight_layout(); plt.savefig(outdir/"phase_plot.png", dpi=180); plt.close()

    # PSD of envelope
    fA, PA = sg.welch(A_H, fs=fsb, nperseg=min(2048,len(A_H)))
    plt.figure(figsize=(9,3)); plt.semilogy(fA, PA)
    plt.axvline(args.fm, ls="--"); plt.xlim(0, max(300, args.lpf)); plt.xlabel("Frequency [Hz]"); plt.ylabel("PSD")
    plt.title(f"{args.event} — Envelope PSD (mark fm≈{args.fm:.1f} Hz)")
    plt.tight_layout(); plt.savefig(outdir/"psd_baseband.png", dpi=180); plt.close()

    # coherence plot
    plt.figure(figsize=(9,3)); plt.plot(fC, Cxy)
    plt.axvline(args.fm, ls="--"); plt.xlim(0, max(300, args.lpf)); plt.ylim(0,1)
    plt.xlabel("Frequency [Hz]"); plt.ylabel("Coherence H1–L1"); plt.title(f"{args.event} — Baseband Coherence")
    plt.tight_layout(); plt.savefig(outdir/"coherence_H1L1.png", dpi=180); plt.close()

    # summary JSON
    summary = {
        "event": args.event,
        "fs_raw": args.fs,
        "fs_baseband": fsb,
        "fc": args.fc,
        "fm": args.fm,
        "preproc": {"bp": args.bp, "whiten": bool(args.whiten), "duration_s": args.duration, "lpf": args.lpf},
        "metrics": {
            "SNR_baseband_H1_at_fm": float(snrA_H),
            "SNR_baseband_L1_at_fm": float(snrA_L),
            "power_baseband_H1_at_fm": float(pwA_H),
            "power_baseband_L1_at_fm": float(pwA_L),
            "coherence_H1L1_at_fm": coh_around_fm,
            "entropy_rate_A_H": ent_H,
            "entropy_rate_A_L": ent_L,
            "mutual_information_bits": MI,
        },
        "outputs": {
            "A_csv": str(outdir/"A_baseband.csv"),
            "phi_csv": str(outdir/"phi_baseband.csv"),
            "fdev_csv": str(outdir/"fdev_baseband.csv"),
            "wav": str(outdir/"baseband.wav"),
            "symbols_csv": str(outdir/"symbols.bin.csv"),
            "plots": {
                "envelope": str(outdir/"envelope_plot.png"),
                "phase": str(outdir/"phase_plot.png"),
                "psd": str(outdir/"psd_baseband.png"),
                "coherence": str(outdir/"coherence_H1L1.png")
            }
        }
    }
    with open(outdir/"info_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[OK] Wrote", outdir)

if __name__ == "__main__":
    main()
