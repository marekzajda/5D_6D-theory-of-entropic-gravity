#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLRT Ω v6+ + QI Gate + SNR (H1/L1[/V1]) — univerzální loader dat

Co umí:
- Volba dat 4 způsoby:
  (A) --profile {gw150914,gw190521,gw191109}
  (B) explicitní cesty: --h1 FILE --l1 FILE [--v1 FILE]
  (C) glob vzory:       --h1 "H-*.txt" --l1 "L-*.txt" ...
  (D) auto-názvy z GWOSC rootu a eventu: --data_root "C:/.../" --event_id "1242442952-32"
      → H-H1_GWOSC_16KHZ_R1-<event>.txt / L-L1_... / V-V1_...
- Bandpass + whiten (Welch), QI gate, π/3 echo, matched-filter SNR, time-slides p-hat
- Paměťově šetrný sken (8 GB VRAM friendly), logy, checkpointy
- Výstupy s časovým razítkem: JSON + CSV

Autor: Marek Zajda + Ω AI copilot
"""

import os, sys, json, math, glob, argparse, time, datetime as dt, traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from scipy.signal.windows import tukey

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

# ------------------------- drobné utilitky -------------------------

def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

def stamp() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def to_tensor(x, device):
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    return torch.tensor(x, dtype=torch.float32, device=device)

def zscore(x: np.ndarray) -> np.ndarray:
    s = np.std(x);  s = 1.0 if (not np.isfinite(s) or s == 0) else s
    return (x - np.mean(x)) / s

def bandpass_sos(fs, f_lo, f_hi, order=6):
    lo = max(1.0, float(f_lo)); hi = float(f_hi)
    return butter(order, [lo/(0.5*fs), hi/(0.5*fs)], btype="band", output="sos")

def bandpass_apply(x: np.ndarray, sos) -> np.ndarray:
    return sosfiltfilt(sos, x).astype(np.float32)

def whiten_welch_cpu(x: np.ndarray, fs: float, nperseg: int = 4096) -> np.ndarray:
    f, Pxx = welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
    if len(Pxx) == 0:
        S = np.ones_like(freqs, dtype=np.float32)
    else:
        S = np.interp(freqs, f, Pxx, left=Pxx[0], right=Pxx[-1])
    S = np.maximum(S, np.median(S)*1e-6)
    Yw = X / np.sqrt(S + 1e-20)
    y = np.fft.irfft(Yw, n=len(x)).astype(np.float32)
    y = (y - y.mean()) / (y.std() + 1e-7)
    return y

def center_crop_to_duration(x: np.ndarray, fs: float, duration_s: float) -> np.ndarray:
    if duration_s is None or duration_s <= 0: return x
    need = int(round(duration_s * fs))
    if len(x) <= need: return x
    start = (len(x) - need) // 2
    return x[start:start+need]

def print_vram():
    if torch.cuda.is_available():
        free,total = torch.cuda.mem_get_info()
        print(f"[VRAM] {free/1e9:.2f}/{total/1e9:.2f} GB free/total")

# --------------------------- datové profily ---------------------------

GW_PROFILES = {
    "gw150914": {
        "h1": "H-H1_GWOSC_16KHZ_R1-1126257414-32.txt",
        "l1": "L-L1_GWOSC_16KHZ_R1-1126257414-32.txt",
        "v1": None
    },
    "gw190521": {
        "h1": "H-H1_GWOSC_16KHZ_R1-1242442952-32.txt",
        "l1": "L-L1_GWOSC_16KHZ_R1-1242442952-32.txt",
        "v1": "V-V1_GWOSC_16KHZ_R1-1242442952-32.txt"
    },
    "gw191109": {
        "h1": "H-H1_GWOSC_16KHZ_R1-1257296840-32.txt",
        "l1": "L-L1_GWOSC_16KHZ_R1-1257296840-32.txt",
        "v1": None
    }
}

# ------------------------------ načítání ------------------------------

def resolve_path(spec: Optional[str]) -> Optional[str]:
    """vrátí konkrétní soubor z cesty NEBO globu; pokud je None, vrátí None"""
    if spec is None: return None
    spec = spec.strip()
    if any(ch in spec for ch in "*?[]"):
        hits = sorted(glob.glob(spec))
        if not hits:
            raise FileNotFoundError(f"No files match pattern: {spec}")
        if len(hits) > 1:
            # preferuj delší název (často obsahuje event id), jinak poslední
            hits = sorted(hits, key=lambda p: (len(os.path.basename(p)), p))
        return hits[-1]
    else:
        if not os.path.exists(spec):
            raise FileNotFoundError(f"File not found: {spec}")
        return spec

def autopath_from_root(root: str, event_id: str, det: str) -> str:
    """
    root + 'H-H1_GWOSC_16KHZ_R1-<event>.txt' styl (det = 'H','L','V').
    """
    det = det.upper()
    if   det == "H": fname = f"H-H1_GWOSC_16KHZ_R1-{event_id}.txt"
    elif det == "L": fname = f"L-L1_GWOSC_16KHZ_R1-{event_id}.txt"
    elif det == "V": fname = f"V-V1_GWOSC_16KHZ_R1-{event_id}.txt"
    else: raise ValueError("det ∈ {H,L,V}")
    path = os.path.join(root, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Auto path not found: {path}")
    return path

def load_timeseries_txt(path: str) -> np.ndarray:
    if path is None: return None
    # podpora .txt / .csv / .gz (pokud text)
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            arr = np.loadtxt(f, dtype=np.float32)
    else:
        arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim > 1: arr = arr[:,0]
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)

@dataclass
class DetData:
    name: str
    t: torch.Tensor
    y: torch.Tensor
    fs: float

def preprocess_to_gpu(arr: np.ndarray, fs: float, f_lo: float, f_hi: float,
                      device: str, whiten: bool, duration: float) -> Tuple[torch.Tensor, torch.Tensor]:
    sos = bandpass_sos(fs, f_lo, f_hi)
    y = bandpass_apply(arr, sos)
    if whiten:
        y = whiten_welch_cpu(y, fs, nperseg=min(4096, len(y)))
    y = center_crop_to_duration(y, fs, duration)
    y = zscore(y) * tukey(len(y), 0.05).astype(np.float32)
    t = np.arange(len(y), dtype=np.float32)/fs
    return to_tensor(t, device), to_tensor(y, device)

# ----------------------------- QI gate -----------------------------

def qi_gate(dets: List[DetData], fs: float, tau_ms: float = 20.0, sharp: float = 1.5) -> None:
    tau = max(1, int(round(fs * tau_ms * 1e-3)))
    ker = np.ones(tau, dtype=np.float32)/tau
    for D in dets:
        env = np.convolve((D.y.detach().cpu().numpy())**2, ker, mode="same")
        m = float(np.max(env)) if np.isfinite(env).all() and env.size else 1.0
        if m <= 0: m = 1.0
        gate = (env/m)**sharp
        G = torch.tensor(np.ascontiguousarray(gate, dtype=np.float32), device=D.y.device)
        if G.numel() != D.y.numel(): G = G[:D.y.numel()]
        D.y = D.y * G

# -------------------------- GLRT komponenty --------------------------

def damped_sine(t: torch.Tensor, f: torch.Tensor, g: torch.Tensor, t0: torch.Tensor):
    tt = t[None,:] - t0[:,None]
    env = torch.where(tt>=0, torch.exp(-g[:,None]*tt), torch.zeros_like(tt))
    ph  = 2*math.pi * f[:,None] * torch.clamp_min(tt, 0.0)
    s   = env * torch.sin(ph)
    c   = env * torch.cos(ph)
    return s, c

def pi_over_3_echo_train(t: torch.Tensor, f: torch.Tensor, g: torch.Tensor, t0: torch.Tensor,
                         dt_echo: torch.Tensor, rho: torch.Tensor, n_echo: int = 4,
                         phase_step: float = math.pi/3.0):
    S0, C0 = damped_sine(t, f, g, t0)
    S = S0.clone(); C = C0.clone()
    ph_acc = 0.0
    for k in range(1, n_echo+1):
        ph_acc += phase_step
        tk = t0 + k*dt_echo
        Sk, Ck = damped_sine(t, f, g, tk)
        cph = math.cos(ph_acc); sph = math.sin(ph_acc)
        gain = (rho**k)[:,None]
        S += ( cph*Sk - sph*Ck) * gain
        C += ( sph*Sk + cph*Ck) * gain
    return S, C

def fit_amp_phase(y: torch.Tensor, S: torch.Tensor, C: torch.Tensor):
    yB = y[None,:]
    SS = torch.sum(S*S, dim=1); CC = torch.sum(C*C, dim=1); SC = torch.sum(S*C, dim=1)
    yS = torch.sum(yB*S, dim=1); yC = torch.sum(yB*C, dim=1)
    det = SS*CC - SC*SC + 1e-12
    a = (yS*CC - yC*SC) / det
    b = (yC*SS - yS*SC) / det
    A = torch.sqrt(a*a + b*b)
    phi = torch.atan2(b, a)
    yhat = a[:,None]*S + b[:,None]*C
    resid2 = (yB - yhat).pow(2)
    return A, phi, resid2

def aic_from_resid(resid2: torch.Tensor, k: int) -> torch.Tensor:
    rss = torch.sum(resid2, dim=1).clamp_min(1e-22)
    n = resid2.shape[1]
    return 2*k + n * torch.log(rss / float(n))

# paměťově úsporný scan
def batched_scan(det: DetData,
                 f_grid: np.ndarray, g_grid: np.ndarray, t0_grid: np.ndarray,
                 batch: int, device: str,
                 use_echo: bool = False,
                 echo_dt_grid: np.ndarray = None,
                 echo_rho_grid: np.ndarray = None,
                 echo_n: int = 4,
                 echo_phase_step: float = math.pi/3.0,
                 prior_dt_mu_ms: float = 1.047,
                 prior_dt_sigma_ms: float = 0.15,
                 prior_dt_weight: float = 0.5) -> Dict:
    t, y = det.t, det.y
    best = {"aic": float("inf")}
    fT, gT, t0T = to_tensor(f_grid, device), to_tensor(g_grid, device), to_tensor(t0_grid, device)
    if use_echo:
        dtT = to_tensor(echo_dt_grid, device)
        rhoT= to_tensor(echo_rho_grid, device)

    total = len(fT)*len(gT)*len(t0T) * ( (len(dtT)*len(rhoT)) if use_echo else 1 )
    processed = 0; t0_bs = max(1, batch//2)

    for f_chunk in fT.split(max(1, batch//2)):
        for g_chunk in gT.split(max(1, batch//2)):
            fgF = f_chunk[:,None].expand(-1,len(g_chunk)).reshape(-1)
            fgG = g_chunk[None,:].expand(len(f_chunk),-1).reshape(-1)
            for t0_chunk in t0T.split(t0_bs):
                f_full = fgF.repeat_interleave(len(t0_chunk))
                g_full = fgG.repeat_interleave(len(t0_chunk))
                t_full = t0_chunk.repeat(len(fgF))
                if not use_echo:
                    cur = f_full.numel()
                    if cur > batch:
                        step = max(1, cur // batch)
                        idx = torch.arange(0, cur, device=device)[::step][:batch]
                        f_sel, g_sel, t_sel = f_full[idx], g_full[idx], t_full[idx]
                    else:
                        f_sel, g_sel, t_sel = f_full, g_full, t_full
                    S, C = damped_sine(t, f_sel, g_sel, t_sel)
                    A, phi, resid2 = fit_amp_phase(y, S, C)
                    aic = aic_from_resid(resid2, k=3)
                    v, i = torch.min(aic, dim=0)
                    if v.item() < best["aic"]:
                        j = i.item()
                        best = dict(aic=float(v.item()),
                                    f=float(f_sel[j].item()),
                                    gamma=float(g_sel[j].item()),
                                    t0=float(t_sel[j].item()),
                                    A=float(A[j].item()),
                                    phi=float(phi[j].item()),
                                    dt_echo=0.0, rho=0.0)
                    processed += f_sel.numel()
                else:
                    for dte in dtT:
                        for rho in rhoT:
                            cur = f_full.numel()
                            if cur > batch:
                                step = max(1, cur // batch)
                                idx = torch.arange(0, cur, device=device)[::step][:batch]
                                f_sel, g_sel, t_sel = f_full[idx], g_full[idx], t_full[idx]
                            else:
                                f_sel, g_sel, t_sel = f_full, g_full, t_full
                            dt_vec = dte.repeat(f_sel.numel())
                            rho_vec= rho.repeat(f_sel.numel())
                            S, C = pi_over_3_echo_train(t, f_sel, g_sel, t_sel, dt_vec, rho_vec, n_echo=echo_n, phase_step=echo_phase_step)
                            A, phi, resid2 = fit_amp_phase(y, S, C)
                            mu  = prior_dt_mu_ms/1000.0
                            sig = max(prior_dt_sigma_ms/1000.0, 1e-6)
                            z   = (dt_vec - mu)/sig
                            resid2 = resid2 + (prior_dt_weight * (z*z))[:,None]
                            aic = aic_from_resid(resid2, k=5)
                            v, i = torch.min(aic, dim=0)
                            if v.item() < best["aic"]:
                                j = i.item()
                                best = dict(aic=float(v.item()),
                                            f=float(f_sel[j].item()),
                                            gamma=float(g_sel[j].item()),
                                            t0=float(t_sel[j].item()),
                                            A=float(A[j].item()),
                                            phi=float(phi[j].item()),
                                            dt_echo=float(dt_vec[j].item()),
                                            rho=float(rho_vec[j].item()))
                            processed += f_sel.numel()

                if processed and (processed % (batch*50) == 0):
                    pct = 100.0*processed/max(1,total)
                    print(f"[SCAN] {processed}/{total} ({pct:4.1f}%) | best AIC={best['aic']:.1f}", end="\r")

    print(f"\n[SCAN] Complete. Best AIC: {best['aic']:.2f}")
    return best

@torch.no_grad()
def snr_matched(y: torch.Tensor, S: torch.Tensor, C: torch.Tensor) -> float:
    yB = y[None,:]
    SS = torch.sum(S*S, dim=1); CC = torch.sum(C*C, dim=1); SC = torch.sum(S*C, dim=1)
    yS = torch.sum(yB*S, dim=1); yC = torch.sum(yB*C, dim=1)
    det = SS*CC - SC*SC + 1e-12
    a = (yS*CC - yC*SC) / det
    b = (yC*SS - yS*SC) / det
    h = a[:,None]*S + b[:,None]*C
    num = torch.sum(yB*h, dim=1).abs()
    den = torch.sqrt(torch.sum(h*h, dim=1) + 1e-22)
    return (num/den)[0].item()

def snr_for_params(det: DetData, pars: Dict, use_echo: bool = False, echo_n: int = 4, echo_phase_step: float = math.pi/3.0) -> float:
    f = torch.tensor([pars["f"]], device=det.y.device)
    g = torch.tensor([pars["gamma"]], device=det.y.device)
    t0= torch.tensor([pars["t0"]], device=det.y.device)
    if use_echo and float(pars.get("dt_echo",0.0))>0.0:
        dt_e = torch.tensor([pars["dt_echo"]], device=det.y.device)
        rho  = torch.tensor([pars["rho"]], device=det.y.device)
        S, C = pi_over_3_echo_train(det.t, f, g, t0, dt_e, rho, n_echo=echo_n, phase_step=echo_phase_step)
    else:
        S, C = damped_sine(det.t, f, g, t0)
    return snr_matched(det.y, S, C)

def network_snr(dets: List[DetData], pars: Dict, use_echo: bool, echo_n: int, echo_phase_step: float):
    per = {D.name: snr_for_params(D, pars, use_echo=use_echo, echo_n=echo_n, echo_phase_step=echo_phase_step) for D in dets}
    rho_net = math.sqrt(sum(v*v for v in per.values()))
    return rho_net, per

def gcc_phat_shift_ms(a: torch.Tensor, b: torch.Tensor, fs: float, max_ms: float = 12.0) -> float:
    N = a.numel()
    Y1 = torch.fft.rfft(a); Y2 = torch.fft.rfft(b)
    R  = Y1 * torch.conj(Y2)
    Rn = R / torch.clamp(torch.abs(R), min=1e-20)
    r  = torch.fft.irfft(Rn, n=N)
    r  = torch.roll(r, N//2)
    lags = torch.arange(-N//2, N//2, device=a.device)
    maxlag = int(round(max_ms*1e-3*fs))
    m = (lags>=-maxlag) & (lags<=maxlag)
    idx = torch.argmax(torch.where(m, r, torch.full_like(r, -1e30)))
    lag = int(idx.item()) - (N//2)
    return 1000.0 * lag / fs

# ------------------------------- MAIN --------------------------------

def main():
    try:
        ap = argparse.ArgumentParser(description="Ω-GLRT v6+ (QI, π/3 echo, SNR) — univerzální načítání H1/L1[/V1]")
        # zdroje dat
        ap.add_argument("--profile", type=str, default=None,
                        help="gw150914|gw190521|gw191109 (nebo nechat prázdné a použít --h1/--l1/--v1 nebo --data_root --event_id)")
        ap.add_argument("--h1", type=str, default=None, help="soubor nebo glob vzor pro H1 (např. H-*.txt)")
        ap.add_argument("--l1", type=str, default=None, help="soubor nebo glob vzor pro L1")
        ap.add_argument("--v1", type=str, default=None, help="soubor nebo glob vzor pro V1 (volitelné)")
        ap.add_argument("--data_root", type=str, default=None, help="kořen složky s GWOSC soubory")
        ap.add_argument("--event_id", type=str, default=None, help="např. 1242442952-32 (použito s --data_root)")
        # zbytek pipeline
        ap.add_argument("--fs", type=float, default=16384.0)
        ap.add_argument("--bp", nargs=2, type=float, default=[60.0, 300.0])
        ap.add_argument("--whiten", action="store_true")
        ap.add_argument("--duration", type=float, default=2.0)
        ap.add_argument("--align", action="store_true")
        ap.add_argument("--batch", type=int, default=4096)
        ap.add_argument("--seed", type=int, default=321)
        ap.add_argument("--outdir", type=str, default=None)
        # echo
        ap.add_argument("--use_echo", action="store_true")
        ap.add_argument("--echo_n", type=int, default=6)
        ap.add_argument("--echo_phase_step", type=float, default=math.pi/3.0)
        ap.add_argument("--echo_dt_ms_lo", type=float, default=0.9)
        ap.add_argument("--echo_dt_ms_hi", type=float, default=1.2)
        ap.add_argument("--echo_dt_n", type=int, default=25)
        ap.add_argument("--echo_rho_lo", type=float, default=0.3)
        ap.add_argument("--echo_rho_hi", type=float, default=0.9)
        ap.add_argument("--echo_rho_n", type=int, default=25)
        ap.add_argument("--prior_dt_sigma_ms", type=float, default=0.15)
        ap.add_argument("--prior_dt_weight", type=float, default=0.5)
        # coarse/hires grid volby
        ap.add_argument("--hires", action="store_true", help="hustší gridy (pomalejší, citlivější)")
        # slides
        ap.add_argument("--slides", type=int, default=0)
        ap.add_argument("--slide_max_shift", type=float, default=0.5)
        # checkpoint (volitelný)
        ap.add_argument("--checkpoint_every", type=int, default=0)
        ap.add_argument("--checkpoint_path", type=str, default=None)

        args = ap.parse_args()
        np.random.seed(args.seed); torch.manual_seed(args.seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try: devname = torch.cuda.get_device_name(0) if device=="cuda" else "CPU"
        except Exception: devname = "CUDA"
        print(f"[{now_utc()}] Device: {devname}")
        print_vram()

        # ---------- Rozhodnutí o datech ----------
        H_path = L_path = V_path = None
        if args.profile:
            prof = args.profile.lower().strip()
            if prof not in GW_PROFILES:
                raise SystemExit(f"Unknown profile: {args.profile}. Known: {list(GW_PROFILES.keys())}")
            H_path = GW_PROFILES[prof]["h1"]
            L_path = GW_PROFILES[prof]["l1"]
            V_path = GW_PROFILES[prof]["v1"]
        elif args.data_root and args.event_id:
            H_path = autopath_from_root(args.data_root, args.event_id, "H")
            L_path = autopath_from_root(args.data_root, args.event_id, "L")
            try:
                V_path = autopath_from_root(args.data_root, args.event_id, "V")
            except FileNotFoundError:
                V_path = None
        else:
            # explicitní cesty / glob vzory
            H_path = resolve_path(args.h1) if args.h1 else None
            L_path = resolve_path(args.l1) if args.l1 else None
            V_path = resolve_path(args.v1) if args.v1 else None

        if not H_path or not L_path:
            raise SystemExit("Provide data via --profile  NEBO  (--h1 & --l1 [--v1])  NEBO  (--data_root & --event_id).")

        print(f"[DATA] H1: {H_path}")
        print(f"[DATA] L1: {L_path}")
        if V_path: print(f"[DATA] V1: {V_path}")

        # ---------- Načtení + předzpracování ----------
        fs = float(args.fs); f_lo, f_hi = [float(x) for x in args.bp]
        h = load_timeseries_txt(H_path);  l = load_timeseries_txt(L_path);  v = load_timeseries_txt(V_path) if V_path else None

        # vyřízni střed okna z dlouhého segmentu (např. 32 s) na zadanou duration
        def center_take(x: np.ndarray, fs: float, dur: float) -> np.ndarray:
            need = int(round(dur*fs))
            if len(x) <= need: return x
            mid = len(x)//2;  return x[mid-need//2: mid+need//2]

        h = center_take(h, fs, max(args.duration, 0.1))
        l = center_take(l, fs, max(args.duration, 0.1))
        if v is not None: v = center_take(v, fs, max(args.duration, 0.1))

        tH,yH = preprocess_to_gpu(h, fs, f_lo, f_hi, device, args.whiten, args.duration)
        tL,yL = preprocess_to_gpu(l, fs, f_lo, f_hi, device, args.whiten, args.duration)
        dets = [DetData("H1", tH, yH, fs), DetData("L1", tL, yL, fs)]
        if v is not None:
            tV,yV = preprocess_to_gpu(v, fs, f_lo, f_hi, device, args.whiten, args.duration)
            dets.append(DetData("V1", tV, yV, fs))

        # ---------- QI gate ----------
        qi_gate(dets, fs, tau_ms=20.0, sharp=1.5)

        # ---------- GCC-PHAT align (L1 -> H1) ----------
        if args.align and len(dets)>=2:
            base, other = dets[0], dets[1]
            dt_ms = gcc_phat_shift_ms(base.y, other.y, fs, max_ms=12.0)
            sh = int(round(dt_ms*1e-3*fs))
            if sh != 0:
                rolled = torch.roll(other.y, shifts=sh)
                if sh > 0: rolled[:sh] = 0.0
                else:      rolled[sh:] = 0.0
                dets[1] = DetData(other.name, other.t, rolled, other.fs)
            print(f"[ALIGN] {other.name} shifted by {dt_ms:+.2f} ms ({sh:+d} samples).")

        # ---------- Gridy ----------
        if args.hires:
            f_grid  = np.linspace(60.0, 300.0, 121).astype(np.float32)
            g_grid  = np.linspace( 2.0, 150.0, 101).astype(np.float32)
            t0_grid = np.linspace(-0.08, 0.12, 61).astype(np.float32)
            if args.echo_dt_n  < 25: args.echo_dt_n  = 25
            if args.echo_rho_n < 25: args.echo_rho_n = 25
            if args.batch < 8192: args.batch = 8192
        else:
            f_grid  = np.linspace(80.0, 200.0, 31).astype(np.float32)
            g_grid  = np.linspace( 5.0, 100.0, 21).astype(np.float32)
            t0_grid = np.linspace(-0.05, 0.10, 16).astype(np.float32)

        if args.use_echo:
            dt_grid  = np.linspace(args.echo_dt_ms_lo/1000.0, args.echo_dt_ms_hi/1000.0, args.echo_dt_n).astype(np.float32)
            rho_grid = np.linspace(args.echo_rho_lo, args.echo_rho_hi, args.echo_rho_n).astype(np.float32)
        else:
            dt_grid  = np.array([0.0], dtype=np.float32)
            rho_grid = np.array([0.0], dtype=np.float32)

        # ---------- Sken na H1 ----------
        print(f"[SCAN] H1 sken (|f|={len(f_grid)},|g|={len(g_grid)},|t0|={len(t0_grid)}, dt|={len(dt_grid)}, rho|={len(rho_grid)}) …")
        seed = batched_scan(
            dets[0], f_grid, g_grid, t0_grid, batch=args.batch, device=("cuda" if torch.cuda.is_available() else "cpu"),
            use_echo=args.use_echo, echo_dt_grid=dt_grid, echo_rho_grid=rho_grid,
            echo_n=args.echo_n, echo_phase_step=args.echo_phase_step,
            prior_dt_mu_ms=1.047, prior_dt_sigma_ms=args.prior_dt_sigma_ms, prior_dt_weight=args.prior_dt_weight
        )
        print(f"[A] Seed on H1: {seed}")

        # ---------- SNR H1/L1(/V1) + síť ----------
        rho_net, rho_per = network_snr(dets, seed, use_echo=args.use_echo, echo_n=args.echo_n, echo_phase_step=args.echo_phase_step)
        print(f"[SNR] " + " ".join([f"{k}={v:.2f}" for k,v in rho_per.items()]) + f" | network={rho_net:.2f}")

        # ---------- Time-slides (rychlý p-hat) ----------
        def p_hat_quick(dets, pars, slides, max_shift_s, seed):
            if slides<=0: return None
            rng = np.random.default_rng(seed)
            rho_obs, _ = network_snr(dets, pars, use_echo=args.use_echo, echo_n=args.echo_n, echo_phase_step=args.echo_phase_step)
            def proj(D, sh_s):
                sh = int(round(sh_s*D.fs))
                y = torch.roll(D.y, shifts=sh)
                if sh>0: y[:sh]=0.0
                elif sh<0: y[sh:]=0.0
                DD = DetData(D.name, D.t, y, D.fs)
                return snr_for_params(DD, pars, use_echo=args.use_echo, echo_n=args.echo_n, echo_phase_step=args.echo_phase_step)
            worse=0
            for _ in range(slides):
                r2=0.0
                for D in dets:
                    r2 += proj(D, rng.uniform(-max_shift_s, max_shift_s))**2
                if math.sqrt(r2) >= rho_obs-1e-9: worse+=1
            return (worse+1.0)/(slides+1.0)

        p_hat = p_hat_quick(dets, seed, slides=args.slides, max_shift_s=args.slide_max_shift, seed=args.seed)
        if p_hat is not None:
            print(f"[BG] p-hat ≈ {p_hat:.4f} (slides={args.slides}, max_shift={args.slide_max_shift}s)")

        # ---------- Výstupy s časovým razítkem ----------
        outdir = args.outdir or f"out_qi_glrt"
        os.makedirs(outdir, exist_ok=True)
        ts = stamp()
        summary = {
            "timestamp_utc": now_utc(),
            "device": devname,
            "inputs": {"H1": H_path, "L1": L_path, "V1": V_path, "fs": fs, "duration": args.duration},
            "bp": {"lo": float(args.bp[0]), "hi": float(args.bp[1])},
            "whiten": bool(args.whiten),
            "align": bool(args.align),
            "qi_gate": {"tau_ms": 20.0, "sharp": 1.5},
            "use_echo": bool(args.use_echo),
            "echo": {
                "n": int(args.echo_n),
                "phase_step": float(args.echo_phase_step),
                "dt_grid_ms": [float(args.echo_dt_ms_lo), float(args.echo_dt_ms_hi), int(args.echo_dt_n)],
                "rho_grid": [float(args.echo_rho_lo), float(args.echo_rho_hi), int(args.echo_rho_n)],
                "prior_dt_ms": 1.047,
                "prior_dt_sigma_ms": float(args.prior_dt_sigma_ms),
                "prior_dt_weight": float(args.prior_dt_weight)
            } if args.use_echo else None,
            "grid_sizes": {"f": len(f_grid), "g": len(g_grid), "t0": len(t0_grid),
                           "dt": (len(dt_grid) if args.use_echo else 0),
                           "rho": (len(rho_grid) if args.use_echo else 0)},
            "batch": int(args.batch),
            "best_on_H1": seed,
            "snr": {"per_detector": rho_per, "network": rho_net},
            "slides": int(args.slides),
            "slide_max_shift_s": float(args.slide_max_shift),
            "p_hat": (None if p_hat is None else float(p_hat))
        }
        out_json = os.path.join(outdir, f"summary_v6plus_qi_glrt_{ts}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[OUT] {out_json}")

        out_csv = os.path.join(outdir, f"snr_per_detector_{ts}.csv")
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("detector,snr\n")
            for k,v in rho_per.items(): f.write(f"{k},{v:.6f}\n")
            f.write(f"network,{rho_net:.6f}\n")
        print(f"[OUT] {out_csv}")

        print("[DONE] Analysis completed successfully!")

    except Exception as e:
        print(f"[ERROR] {e}")
        print(f"[TRACE] {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
