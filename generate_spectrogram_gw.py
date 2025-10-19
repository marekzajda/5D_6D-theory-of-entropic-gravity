#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate time–frequency spectrograms for GW150914 (or any event) from raw strain files.
Usage:
python generate_spectrogram_gw.py --h1 H-H1_GWOSC_16KHZ_R1-1126259447-32.txt --l1 L-L1_GWOSC_16KHZ_R1-1126259447-32.txt --fs 16384 --event GW150914
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_strain_txt(path: Path):
    arr = np.loadtxt(str(path))
    if arr.ndim == 1:
        return None, arr.astype(float)
    if arr.shape[1] == 1:
        return None, arr[:,0].astype(float)
    return arr[:,0].astype(float), arr[:,1].astype(float)

def center_crop(x, fs, dur=2.0):
    if dur is None or dur <= 0: return x
    N = int(round(dur*fs))
    if len(x) <= N: return x
    mid = len(x)//2
    return x[mid-N//2:mid+N//2]

def plot_spec(h, fs, title, overlays=None, outpath="spec.png"):
    plt.figure(figsize=(8,4))
    NFFT = 1024
    nover = NFFT//2
    Pxx, freqs, bins, im = plt.specgram(h, NFFT=NFFT, Fs=fs, noverlap=nover, cmap=None)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(title)
    if overlays:
        for f, lbl in overlays:
            if 0 < f < fs/2:
                plt.axhline(f, linestyle="--", linewidth=1)
                plt.text(bins[-1]*0.98, f, f"{lbl}", va="bottom", ha="right", fontsize=8, rotation=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1", required=True, type=Path)
    ap.add_argument("--l1", required=True, type=Path)
    ap.add_argument("--fs", type=float, default=16384.0)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--event", type=str, default="GWXXXXXX")
    ap.add_argument("--fc", type=float, default=955.0)
    ap.add_argument("--fm", type=float, default=141.7)
    args = ap.parse_args()

    tH, hH = load_strain_txt(args.h1); tL, hL = load_strain_txt(args.l1)
    hH = center_crop(hH, args.fs, args.duration)
    hL = center_crop(hL, args.fs, args.duration)

    overlays = [
        (args.fm, "f_m ≈ %.1f Hz" % args.fm),
        (args.fc-args.fm, "f_c - f_m"),
        (args.fc, "f_c ≈ %.0f Hz" % args.fc),
        (args.fc+args.fm, "f_c + f_m"),
        (args.fc/7.0, "f_c/7"),
        (args.fc/8.0, "f_c/8"),
        (2*args.fc, "2 f_c"),
    ]

    outH = Path(f"{args.event}_H1_spectrogram.png")
    outL = Path(f"{args.event}_L1_spectrogram.png")

    plot_spec(hH, args.fs, f"{args.event} — H1 Spectrogram", overlays=overlays, outpath=outH)
    plot_spec(hL, args.fs, f"{args.event} — L1 Spectrogram", overlays=overlays, outpath=outL)

    print(f"[OK] Wrote {outH} and {outL}")

if __name__ == "__main__":
    main()
