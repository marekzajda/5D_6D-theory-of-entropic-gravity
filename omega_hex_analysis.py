#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Omega HEX (π/3) phase analysis
------------------------------
- Vstup: jeden nebo více eventů (adresáře OUT_INFO/<EVENT>/), kde:
    * A_baseband.csv : "t, A_H, A_L"  (obálka po demodulaci nosné ~955 Hz)
    * (volitelné) phi_baseband.csv : "t, phi_H, phi_L"  (fázový kanál po demodu)
    * (volitelné) symbols.bin_<EVENT>.csv : "bit_index,bit" (řádky bez ',1' se berou jako 0)

- Postup:
    1) Vypočítá fázi modulace φ(t) kolem fm (141.7 Hz):
       - preferuje "phi_baseband.csv"; jinak vezme A(t), zúží pásmo 100–200 Hz, a spočítá analytický signál (Hilbert) → φ(t).
    2) Kvantizace φ(t) mod 2π do 6 košů (π/3):
           S(t) ∈ {0,1,2,3,4,5}
    3) Metriky:
       - occupancy 6 košů, Shannon H, přechodová matice T(6x6), sousední přechody (i→i±1),
         autokorelace stavů (periodicita ~6–7 bitů), „hex-PSK“ koherence.
    4) Grafy:
       - rose-plot (polar histogram), přechodová matice (heatmap), autokorelace stavů,
         rozdělení φ mod 2π, histogram S(t).
    5) Multi-event:
       - Jensen–Shannon divergence mezi occupancy i mezi přechodovými maticemi,
         srovnávací tabulka (.tex) a JSON report.

Použití (příklady):
    python omega_hex_analysis.py --events OUT_INFO/GW151226 OUT_INFO/GW170608 OUT_INFO/GW200220 \
        --fm 141.7 --bp 100 200 --save plots_hex --tex hex_metrics.tex --json hex_metrics.json

    # s přímým dosazením CSV se symboly:
    python omega_hex_analysis.py --events OUT_INFO/GW151226 --symbols symbols.bin_151226.csv
"""

import argparse, json, os, math
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as sg
import matplotlib.pyplot as plt

# --------------------------- I/O helpers ---------------------------

def load_baseband(env_dir: Path):
    """Načti A_baseband.csv (t, A_H, A_L); volitelně phi_baseband.csv (t, phi_H, phi_L)."""
    A_path = env_dir / "A_baseband.csv"
    phi_path = env_dir / "phi_baseband.csv"
    if not A_path.exists() and not phi_path.exists():
        raise FileNotFoundError(f"Nevidím A_baseband.csv ani phi_baseband.csv v {env_dir}")

    t, AH, AL = None, None, None
    if A_path.exists():
        A = np.loadtxt(A_path, delimiter=",", skiprows=1)
        t = A[:,0]; AH = A[:,1]; AL = A[:,2]
    phi_t = phi_H = phi_L = None
    if phi_path.exists():
        P = np.loadtxt(phi_path, delimiter=",", skiprows=1)
        phi_t = P[:,0]; phi_H = P[:,1]; phi_L = P[:,2]
    return (t, AH, AL), (phi_t, phi_H, phi_L)

def load_symbols_csv(path: Path):
    """Načte CSV 'bit_index,bit'; řádek bez ',1' znamená bit=0. Vrací (idx[], bits[]) souvisle vyplněné."""
    if path is None or not path.exists(): return None, None
    idx, bits = [], []
    with open(path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts)==1:
                try: i = int(parts[0])
                except: continue
                b = 0
            else:
                try: i = int(parts[0])
                except: continue
                try: b = int(parts[1])
                except: b = 0
            idx.append(i); bits.append(b)
    if not idx: return None, None
    order = np.argsort(idx)
    idx = np.array(idx, int)[order]
    bits = np.array(bits, int)[order]
    full_idx = np.arange(idx.min(), idx.max()+1, dtype=int)
    pos = {i:b for i,b in zip(idx, bits)}
    full_bits = np.array([pos.get(i,0) for i in full_idx], dtype=int)
    return full_idx, full_bits

# --------------------------- Signal utils ---------------------------

def bandpass(x, fs, lo, hi, order=4):
    lo = max(0.5, lo); hi = min(fs*0.49, hi)
    sos = sg.butter(order, [lo/(0.5*fs), hi/(0.5*fs)], btype="band", output="sos")
    return sg.sosfiltfilt(sos, x)

def analytic_phase_from_env(A, fs, lo=100.0, hi=200.0):
    """Zúží pásmo obálky okolo fm a vezme Hilbert fázi."""
    x = bandpass(A, fs, lo, hi, order=4)
    z = sg.hilbert(x)
    phi = np.angle(z)  # [-pi, pi)
    return phi

def unwrap_mod2pi(phi):
    """Vrátí φ mod 2π v [0, 2π)."""
    return (phi + 2*np.pi) % (2*np.pi)

def quantize_pi_over_3(phi_mod):
    """Kvantizace φ ∈ [0,2π) do 6 košů po π/3: S∈{0..5}."""
    bins = np.linspace(0, 2*np.pi, 7)  # 0, π/3, ..., 2π
    S = np.digitize(phi_mod, bins) - 1
    S[S==6] = 5
    return S

def occupancy(S):
    hist = np.bincount(S, minlength=6).astype(float)
    p = hist / hist.sum() if hist.sum()>0 else hist
    H = -(p[p>0]*np.log2(p[p>0])).sum() if p.sum()>0 else 0.0
    return p, H

def transition_matrix(S):
    M = np.zeros((6,6), dtype=float)
    for i in range(len(S)-1):
        a, b = int(S[i]), int(S[i+1])
        if 0<=a<6 and 0<=b<6:
            M[a,b]+=1.0
    if M.sum()>0:
        M = M / np.maximum(M.sum(axis=1, keepdims=True), 1e-12)
    return M

def js_divergence(p, q):
    """Jensen-Shannon divergence mezi dvěma pravděpodobnostními vektory (bezpečně)."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p/np.sum(p) if np.sum(p)>0 else p
    q = q/np.sum(q) if np.sum(q)>0 else q
    m = 0.5*(p+q)
    def _kl(a,b):
        mask = (a>0) & (b>0)
        return np.sum(a[mask]*np.log2(a[mask]/b[mask]))
    return 0.5*_kl(p,m) + 0.5*_kl(q,m)

def state_autocorr(S, maxlag=64):
    x = S.astype(float)
    x = (x - x.mean()) / (x.std()+1e-9)
    r = np.correlate(x, x, mode="full")
    r = r[len(x)-1:len(x)-1+maxlag]
    r = r / (r[0] if r[0]!=0 else 1.0)
    return r

# --------------------------- Plotting ---------------------------

def plot_polar_rose(phi_mod, out_png, title="Rose plot (φ mod 2π)"):
    th = phi_mod
    n=36
    hist, edges = np.histogram(th, bins=n, range=(0,2*np.pi))
    centers = 0.5*(edges[:-1]+edges[1:])
    ax = plt.subplot(111, projection='polar')
    ax.bar(centers, hist, width=(2*np.pi/n), align='center')
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def plot_state_hist(S, out_png, title="State occupancy (S∈{0..5})"):
    plt.figure(figsize=(4.5,3))
    counts = np.bincount(S, minlength=6)
    plt.bar(np.arange(6), counts)
    plt.xlabel("State (0..5)"); plt.ylabel("Count"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def plot_transition_heatmap(M, out_png, title="Transition matrix (6×6)"):
    plt.figure(figsize=(4.6,4.2))
    plt.imshow(M, cmap="viridis", vmin=0, vmax=np.max(M)+1e-9)
    plt.colorbar(label="P(next=j | current=i)")
    plt.xlabel("j"); plt.ylabel("i"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def plot_autocorr(r, out_png, title="Autocorrelation of S(t)"):
    lags = np.arange(len(r))
    plt.figure(figsize=(5,3))
    try:
        plt.stem(lags, r, basefmt=" ", markerfmt="o", linefmt="-")
    except TypeError:
        # starší verze matplotlibu
        plt.stem(lags, r)
    plt.xlabel("Lag [samples]")
    plt.ylabel("ρ")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_phi_hist(phi_mod, out_png, title="φ mod 2π"):
    plt.figure(figsize=(5,3))
    plt.hist(phi_mod, bins=36, range=(0,2*np.pi))
    plt.xlabel("phase [rad]"); plt.ylabel("count"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

# --------------------------- Main per-event ---------------------------

def analyse_event(event_dir: Path, fm=141.7, bp=(100.0,200.0), fs_baseband=None,
                  symbols_csv: Path=None, out_dir: Path=None, name=None):
    (t, AH, AL), (pt, pH, pL) = load_baseband(event_dir)
    # odhad fs z času
    if t is not None:
        fs = 1.0/float(t[1]-t[0])
    elif pt is not None:
        fs = 1.0/float(pt[1]-pt[0])
    else:
        fs = 2048.0 if fs_baseband is None else fs_baseband

    # zvol stream pro fázi: preferuj phi_baseband, jinak Hilbert nad A_NET
    if pH is not None and pL is not None:
        phi_net = np.unwrap(np.angle(np.exp(1j*pH)) + np.angle(np.exp(1j*pL)))/2.0
        phi_mod = unwrap_mod2pi(phi_net)
    else:
        # A_NET
        if AH is None or AL is None:
            raise RuntimeError("Chybí A_baseband.csv i phi_baseband.csv.")
        A_net = 0.5*(AH + AL)
        # pásmo kolem fm
        A_bp = bandpass(A_net, fs, bp[0], bp[1], order=4)
        phi_mod = unwrap_mod2pi(np.angle(sg.hilbert(A_bp)))

    # kvantizace na π/3
    S = quantize_pi_over_3(phi_mod)

    # metriky
    p6, H = occupancy(S)
    M = transition_matrix(S)
    r = state_autocorr(S, maxlag=64)
    neigh_prob = 0.0
    if M.sum()>0:
        # průměrná pravděpodobnost přechodu do souseda (i->i±1 mod 6)
        for i in range(6):
            neigh_prob += 0.5*(M[i, (i+1)%6] + M[i, (i-1)%6])
        neigh_prob /= 6.0

    # symboly (volitelné) a 6×4 bloky
    blocks_bits = None
    if symbols_csv is not None and symbols_csv.exists():
        idx, bits = load_symbols_csv(symbols_csv)
        if bits is not None:
            # přesně 24 → 4×6
            if len(bits) >= 24:
                bits24 = bits[:24]
            else:
                bits24 = np.zeros(24, dtype=int); bits24[:len(bits)] = bits
            blocks_bits = bits24.reshape(4,6)

    # výstupy
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = name or event_dir.name
        plot_polar_rose(phi_mod, out_dir/f"{tag}_rose.png", title=f"{tag} — Rose plot (φ mod 2π)")
        plot_phi_hist(phi_mod, out_dir/f"{tag}_phi_hist.png", title=f"{tag} — φ mod 2π")
        plot_state_hist(S, out_dir/f"{tag}_state_hist.png", title=f"{tag} — State occupancy")
        plot_transition_heatmap(M, out_dir/f"{tag}_transitions.png", title=f"{tag} — Transition matrix")
        plot_autocorr(r, out_dir/f"{tag}_autocorr.png", title=f"{tag} — Autocorr S(t)")

    info = {
        "event": name or event_dir.name,
        "fs": fs,
        "fm": fm,
        "bandpass": list(bp),
        "entropy_bits": float(H),
        "occupancy": p6.tolist(),
        "neighbor_transition_prob": float(neigh_prob),
        "transition_matrix": M.tolist(),
        "autocorr_first_peaks": int(np.argmax(r[1:])+1) if len(r)>1 else None,
        "S_length": int(len(S)),
        "has_symbols": bool(blocks_bits is not None),
        "blocks_bits": blocks_bits.tolist() if blocks_bits is not None else None
    }
    return info

# --------------------------- Multi-event orchestration ---------------------------

def js_divergence_matrices(M1, M2):
    """JS divergence mezi řádky matic 6x6, průměr přes počáteční stavy."""
    M1 = np.asarray(M1, float); M2 = np.asarray(M2, float)
    divs = []
    for i in range(6):
        d = js_divergence(M1[i,:]+1e-12, M2[i,:]+1e-12)
        divs.append(d)
    return float(np.mean(divs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", type=Path, required=True, help="Jedna či více složek OUT_INFO/<EVENT>/")
    ap.add_argument("--symbols", nargs="*", type=Path, default=[], help="CSV se symboly (bit_index,bit) – mapují se podle pořadí k events")
    ap.add_argument("--fm", type=float, default=141.7)
    ap.add_argument("--bp", nargs=2, type=float, default=[100.0, 200.0], help="Bandpass pro modulaci [Hz]")
    ap.add_argument("--save", type=Path, default=Path("HEX_PLOTS"))
    ap.add_argument("--tex", type=Path, default=Path("hex_metrics.tex"))
    ap.add_argument("--json", type=Path, default=Path("hex_metrics.json"))
    args = ap.parse_args()

    args.save.mkdir(parents=True, exist_ok=True)
    results = []

    # Sladění symbolových souborů (pokud jsou) k eventům podle indexu
    sym_map = {}
    for i, ev in enumerate(args.events):
        if i < len(args.symbols):
            sym_map[ev] = args.symbols[i]
        else:
            sym_map[ev] = None

    # Jednotlivé analýzy
    for i, ev in enumerate(args.events):
        info = analyse_event(
            ev, fm=args.fm, bp=(args.bp[0], args.bp[1]),
            symbols_csv=sym_map.get(ev), out_dir=args.save, name=ev.name
        )
        results.append(info)

    # Multi-event JS divergence (occupancy & transitions)
    comparisons = []
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            A, B = results[i], results[j]
            d_occ = js_divergence(np.array(A["occupancy"]), np.array(B["occupancy"]))
            d_mat = js_divergence_matrices(np.array(A["transition_matrix"]), np.array(B["transition_matrix"]))
            comparisons.append({
                "A": A["event"], "B": B["event"],
                "JS_div_occupancy": d_occ,
                "JS_div_transition": d_mat
            })

    # Ulož JSON
    out = {"events": results, "comparisons": comparisons}
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Vytvoř LaTeX tabulku
    # (event, H, neighbor_transition_prob, peak_lag, occ[0..5])
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Event & $H$ [bits] & $p_{\mathrm{neigh}}$ & peak lag & $p_0$ & $p_1$ & $p_2$ & $p_3$ & $p_4$ & $p_5$ \\")
    lines.append(r"\midrule")
    for R in results:
        occ = R["occupancy"]
        line = f"{R['event']} & {R['entropy_bits']:.3f} & {R['neighbor_transition_prob']:.3f} & {R['autocorr_first_peaks']} & " + \
               " & ".join(f"{p:.2f}" for p in occ)
        lines.append(line + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Ω-hex (π/3) metrics: Shannon entropy $H$, průměrná pravděpodobnost sousedních přechodů $p_{\mathrm{neigh}}$, první autokorelační pík (lag v samples) a obsazenosti stavů $p_{0..5}$.}")
    lines.append(r"\label{tab:omega_hex_metrics}")
    lines.append(r"\end{table}")
    with open(args.tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Uloženo: plots→{args.save}, JSON→{args.json}, LaTeX→{args.tex}")

if __name__ == "__main__":
    main()
