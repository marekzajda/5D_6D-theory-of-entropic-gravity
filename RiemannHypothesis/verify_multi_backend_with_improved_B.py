# verify_multi_backend_with_improved_B.py
# Purpose: Cross-verify the GPN falsification pipeline with alternative zeta backends
#          (RS and Approximate Functional Equation), and improved background B(t).
# Code in English; Czech comments explain design and math choices.

import numpy as np
import mpmath as mp
from pathlib import Path

# ======================== user parameters =========================
# --- Global analysis window ---
MP_DPS    = 100            # mpmath precision (vyšší = pomalejší)
CENTER_T  = 1.0e8          # center of the time window (e.g., 1e8, 1e9)
WINDOW    = 2000.0         # window width in t
POINTS    = 4096           # number of time samples (derivatives/FFT resolution)

# --- Backend selection: "rs" | "afe" | "external" ---
BACKEND   = "afe"           # RS=Riemann–Siegel; AFE=Approx. Functional Equation; external=data files

# If BACKEND == "external", provide:
EXTERNAL_CSV = "data.csv"  # CSV with "t,logabsxi" OR "t,logabszeta" (equidistant t)
EXTERNAL_ZEROS = "zeros.txt"  # TXT with zeros t_k (one per line; wider than analysis window)

# --- Zeros detection around window (for RS/AFE) ---
ZERO_OVERSAMPLE  = 8       # dense zero grid ~oversample * mean spacing
ZERO_MARGIN_FRAC = 0.50    # extend detection range beyond window (fractions of WINDOW)

# --- Poisson & cleanup parameters ---
A_LIST    = [3.0, 2.0, 1.0, 0.5]  # Poisson parameters
L_LIST    = [1.0, 0.5, 1/3, 0.25] # local Gaussian mollifiers
EPSILON   = 1e-4                   # informational threshold for core-max
IGNORE_EDGE_FRAC   = 0.35          # ignore 30% edges when measuring metrics
PAD_FACTOR         = 12             # FFT zero-padding factor
LOWFREQ_CYCLES_CUT = 12.0           # remove |ω| < 2π * cycles / WINDOW
KILL_LOW_BINS      = 6             # additionally kill a few lowest bins
CORE_POLY_DETREND_ORDER = 4        # polynomial detrend (0..3) fitted on core
TAPER_ALPHA        = 0.40          # Tukey apodization in time (0=none, 1=Hann)

# --- Improved background B(t) options ---
B_METHOD    = "auto"  # "auto" -> asymptotic ψ1(z) for large |t|, exact for small |t|
B_T_SWITCH  = 2.0e4   # threshold |t| above which to use asymptotic series
B_TERMS     = 8       # number of Bernoulli-series terms in asymptotic ψ1 (>=3 recommended)

# --- AFE backend controls ---
AFE_T_BLOCK = 128     # t-block size for vectorized AFE summation
# N cutoff is chosen ~ sqrt(t/(2π)) per standard heuristic; we compute per-center.


# ======================== utilities: grids, FFT, windows =========================

def make_t_grid(center, window, points):
    t0 = center - 0.5*window
    t1 = center + 0.5*window
    t = np.linspace(t0, t1, int(points))
    return t, float(t[1] - t[0])

def omega_axis(dt, N):
    return 2*np.pi*np.fft.fftfreq(N, d=dt)

def d2dt_spectral(f_vals, dt):
    # 2nd derivative via spectral filter (stabilní; přesné)
    N = len(f_vals)
    om = omega_axis(dt, N)
    F = np.fft.fft(f_vals)
    return np.real(np.fft.ifft(-(om**2) * F))

def tukey_window(N, alpha=0.25):
    # apodizace okna -> menší okrajové úniky (leakage)
    if alpha <= 0: return np.ones(N)
    if alpha >= 1: return np.hanning(N)
    w = np.ones(N)
    L = int(np.floor(alpha*(N-1)/2.0))
    if L <= 0: return w
    x = np.linspace(0, 1, L, endpoint=False)
    left  = 0.5*(1 + np.cos(np.pi*(2*x/alpha - 1)))
    right = left[::-1]
    w[:L]  = left
    w[-L:] = right
    return w

def kappa_gaussian(L, dx=1.0):
    # normalizovaný Gauss pro lokální mollifikaci
    M = max(7, int(6 * L / max(dx, 1e-12)))
    if M % 2 == 0: M += 1
    x = np.linspace(-3*L, 3*L, M)
    w = np.exp(-(x**2)/(2*L**2))
    w /= np.sum(w)
    return w


# ======================== background: B(t) improved =========================
# B(t) = -1/4 Re ψ1(1/4 + i t/2)

# Bernoulli numbers B2, B4, ..., used in asymptotic series for trigamma ψ1
_BERNOULLI_EVEN = {
    2:  1.0/6.0,
    4: -1.0/30.0,
    6:  1.0/42.0,
    8: -1.0/30.0,
    10: 5.0/66.0,
    12: -691.0/2730.0,
    14: 7.0/6.0,
    16: -3617.0/510.0,
    # víc by šlo doplnit; pro B_TERMS<=8 je tohle víc než dost
}

def trigamma_asym(z, terms=6):
    # asymptotika: ψ1(z) ≈ 1/z + 1/(2 z^2) + Σ_{k≥1} B_{2k} / z^{2k+1}
    inv = 1.0/z
    s  = inv + 0.5*inv*inv
    k_used = 0
    for k in range(1, 100):
        if 2*k not in _BERNOULLI_EVEN: break
        s += _BERNOULLI_EVEN[2*k] * (inv ** (2*k+1))
        k_used += 1
        if k_used >= terms:
            break
    return s

def background_curvature_improved(t_vec, method="auto", t_switch=2e4, terms=6):
    # Přesný výpočet u menších t (mpmath polygamma), asymptotika u velkých t (rychlé; stabilní).
    mp.mp.dps = MP_DPS
    out = np.empty_like(t_vec, dtype=float)
    for i, t in enumerate(t_vec):
        z = mp.mpf('0.25') + 0.5j*mp.mpf(t)
        if method == "exact" or (method == "auto" and abs(t) <= t_switch):
            psi1 = mp.polygamma(1, z)
        else:
            psi1 = trigamma_asym(z, terms=terms)
        out[i] = -0.25 * float(mp.re(psi1))
    return out


# ======================== θ(t) and RS core (for RS and AFE Z) =========================

def theta_RS(t):
    # Hardy's theta via loggamma (funguje i pro velká t)
    z = mp.mpf('0.25') + 0.5j*mp.mpf(t)
    return float(mp.im(mp.loggamma(z)) - 0.5*mp.mpf(t)*mp.log(mp.pi))

def precompute_RS(center_t):
    # RS cutoff N ~ sqrt(T/(2π))
    N = int(np.floor(np.sqrt(float(center_t)/(2.0*np.pi))))
    if N < 50:
        raise ValueError("RS cutoff N too small; increase CENTER_T.")
    n = np.arange(1, N+1, dtype=np.float64)
    return N, np.log(n), n**(-0.5)

def Z_RS(t_vec, N, logn, w, block=4000):
    # Hardy Z(t) z RS hlavní sumy (blokově)
    theta = np.array([theta_RS(float(t)) for t in t_vec], dtype=float)
    Z = np.zeros_like(theta)
    m = len(logn)
    for start in range(0, m, block):
        end = min(start+block, m)
        ph = np.outer(t_vec, logn[start:end]) - theta[:, None]
        Z += 2.0 * (np.cos(ph) @ w[start:end])
    return Z

def mean_zero_spacing(T):
    return 2.0*np.pi / np.log(T/(2.0*np.pi))


# ======================== Zeros detection (sign changes of Z) =========================

def detect_zeros_from_Z(t_vec, Z_vals, refine=True, refine_steps=14, backend="rs"):
    # najde změny znaménka a zjemní bisekcí (volá backendovou Z-evaluaci po bodech)
    idx = np.where(np.sign(Z_vals[:-1])*np.sign(Z_vals[1:]) < 0)[0]
    if not refine or len(idx) == 0:
        return 0.5*(t_vec[idx] + t_vec[idx+1])

    zeros = []
    # Backend-specific re-eval function for a single t
    if backend == "rs":
        N, logn, w = precompute_RS(0.5*(t_vec[0]+t_vec[-1]))
        def Z_one(t):
            return Z_RS(np.array([t], dtype=float), N, logn, w)[0]
    elif backend == "afe":
        # for AFE refine, recompute ζ then Z=Re(e^{iθ} ζ)
        Ncut = int(np.floor(np.sqrt(0.5*(t_vec[0]+t_vec[-1])/(2.0*np.pi))))
        logn = np.log(np.arange(1, Ncut+1, dtype=np.float64))
        w    = np.arange(1, Ncut+1, dtype=np.float64)**(-0.5)
        def zeta_afe_one(t):
            s = 0.5 + 1j*t
            # S1 = Σ n^{-1/2-it} = w * e^{-i t log n}
            S1 = np.dot(w, np.exp(-1j*t*logn))
            # χ(s)
            chi = complex(mp.power(mp.pi, s-0.5) * mp.gamma((1.0-s)/2.0) / mp.gamma(s/2.0))
            # S2 = Σ n^{-1/2+it} = w * e^{+i t log n}
            S2 = np.dot(w, np.exp(+1j*t*logn))
            return S1 + chi*S2
        def Z_one(t):
            return float(np.real(np.exp(1j*theta_RS(float(t))) * zeta_afe_one(float(t))))
    else:
        raise ValueError("Unknown backend for refine.")
    # bisection
    for a, b in zip(t_vec[idx], t_vec[idx+1]):
        aa, bb = float(a), float(b)
        Za = Z_one(aa); Zb = Z_one(bb)
        for _ in range(refine_steps):
            m = 0.5*(aa+bb)
            Zm = Z_one(m)
            if Za*Zm <= 0.0:
                bb, Zb = m, Zm
            else:
                aa, Za = m, Zm
        zeros.append(0.5*(aa+bb))
    return np.array(zeros, dtype=float)

def detect_zeros_dense(center_T, t_lo, t_hi, oversample, backend="rs"):
    # postaví hustou mřížku ~oversample * mean spacing a zavolá backend Z
    spacing = float(mean_zero_spacing(center_T))
    dtz = spacing / max(oversample, 2)
    t_grid = np.arange(t_lo, t_hi + 0.5*dtz, dtz, dtype=float)

    if backend == "rs":
        N, logn, w = precompute_RS(center_T)
        Zv = Z_RS(t_grid, N, logn, w)
        zeros = detect_zeros_from_Z(t_grid, Zv, refine=True, backend="rs")
    elif backend == "afe":
        # AFE Z(t): Z = Re(e^{iθ} ζ(s)), ζ(s) via AFE block summation
        ζ = zeta_afe_vec(t_grid)         # complex array
        Z = np.real(np.exp(1j*np.array([theta_RS(float(t)) for t in t_grid])) * ζ)
        zeros = detect_zeros_from_Z(t_grid, Z, refine=True, backend="afe")
    else:
        raise ValueError("Backend must be 'rs' or 'afe' for detection.")
    return t_grid, zeros


# ======================== AFE backend: ζ(1/2+it) vectorized =========================

def zeta_afe_vec(t_vec):
    """
    Vectorized Approximate Functional Equation at s=1/2+it.
    ζ(s) ≈ Σ_{n≤N} n^{-s} + χ(s) Σ_{n≤N} n^{s-1}, N ~ sqrt(t/(2π)).
    Vypočítáno blokově přes t pro paměťovou úsporu.
    """
    mp.mp.dps = MP_DPS
    t_vec = np.array(t_vec, dtype=float)
    # choose N from center T (good enough within the window)
    Ncut = int(np.floor(np.sqrt(0.5*(t_vec[0]+t_vec[-1])/(2.0*np.pi))))
    if Ncut < 10:
        raise ValueError("AFE cutoff too small; increase CENTER_T.")
    n = np.arange(1, Ncut+1, dtype=np.float64)
    logn = np.log(n)
    w = n**(-0.5)

    # block over t to limit memory
    out = np.empty(t_vec.shape, dtype=np.complex128)
    B = AFE_T_BLOCK
    for i in range(0, len(t_vec), B):
        tb = t_vec[i:i+B]
        # S1 = Σ w * e^{-i t log n}
        phase1 = np.exp(-1j * np.outer(tb, logn))  # shape (len(tb), Ncut)
        S1 = phase1 @ w
        # S2 = Σ w * e^{+i t log n}
        # (konjugovaná fáze = e^{+i t log n})
        S2 = np.conj(S1)  # využij symetrie; přesné protože w je reálné
        # χ(s) per t
        chi = np.array([complex(mp.power(mp.pi, 0.5+1j*tt - 0.5)
                                * mp.gamma((1.0-(0.5+1j*tt))/2.0)
                                / mp.gamma((0.5+1j*tt)/2.0))
                        for tt in tb], dtype=np.complex128)
        out[i:i+B] = S1 + chi * S2
    return out


# ======================== Φcrit and external loader =========================

def phi_crit_from_backend_RS(t):
    # Φcrit(t) = log|xi| přes RS: log|Z| + Re(logΓ(s/2)-(s/2)logπ) + log|s(s-1)/2|
    mp.mp.dps = MP_DPS
    N, logn, w = precompute_RS(0.5*(t[0]+t[-1]))
    Z = Z_RS(t, N, logn, w)
    logabsz = np.log(np.maximum(np.abs(Z), 1e-300))
    phi = np.array(logabsz, dtype=float)
    half = mp.mpf('0.5')
    for i, ti in enumerate(t):
        s = half + 1j*mp.mpf(ti)
        bg = mp.loggamma(s/2.0) - (s/2.0)*mp.log(mp.pi)
        phi[i] += float(mp.re(bg))
        ss = s*(s-1.0)/2.0
        phi[i] += float(mp.log(abs(ss)))
    return phi

def phi_crit_from_backend_AFE(t):
    # Φcrit(t) = log|xi| přes AFE: vezmi ζ(s) z AFE a dopočti Γ/π část
    mp.mp.dps = MP_DPS
    zeta = zeta_afe_vec(t)
    logabsz = np.log(np.maximum(np.abs(zeta), 1e-300))
    phi = np.array(logabsz, dtype=float)
    half = mp.mpf('0.5')
    for i, ti in enumerate(t):
        s = half + 1j*mp.mpf(ti)
        bg = mp.loggamma(s/2.0) - (s/2.0)*mp.log(mp.pi)
        phi[i] += float(mp.re(bg))
        ss = s*(s-1.0)/2.0
        phi[i] += float(mp.log(abs(ss)))
    return phi

def load_external_series(csv_path):
    arr = np.loadtxt(csv_path, delimiter=",", ndmin=2)
    if arr.shape[1] < 2:
        raise ValueError("CSV must have two columns: t, value (log|xi| or log|zeta|)")
    t = arr[:,0].astype(float); y = arr[:,1].astype(float)
    return t, y

def phi_crit_from_external(t, y):
    # Pokud y = log|zeta|, přičteme Γ/π a algebraickou část → log|xi|; pokud už je log|xi|, přičtení přidá jen konstantu → 2. derivace ji vyruší.
    mp.mp.dps = MP_DPS
    phi = np.array(y, dtype=float)
    half = mp.mpf('0.5')
    for i, ti in enumerate(t):
        s = half + 1j*mp.mpf(ti)
        bg = mp.loggamma(s/2.0) - (s/2.0)*mp.log(mp.pi)
        phi[i] += float(mp.re(bg))
        ss = s*(s-1.0)/2.0
        phi[i] += float(mp.log(abs(ss)))
    return phi

def load_zeros(path):
    z = np.loadtxt(path, ndmin=1).astype(float)
    return np.sort(z)


# ======================== Poisson + spectral ops =========================

def poisson_smooth_fft_padded(signal, dt, a, pad_factor=8, taper_alpha=0.0):
    # Poisson smoothing in frequency with center-padding and optional Tukey apodization
    N = len(signal)
    if taper_alpha > 0:
        tap = tukey_window(N, alpha=taper_alpha)
        signal = signal * tap
    M = int(pad_factor * N)
    left = (M - N)//2; right = M - N - left
    sig_pad = np.pad(signal, (left, right), mode='constant', constant_values=0.0)
    om = omega_axis(dt, M)
    H  = np.fft.fft(sig_pad)
    Hs = np.exp(-a*np.abs(om)) * H
    return Hs, om, left, right, M

def project_out_online_component(Hs, om, t_zeros, rcond=1e-4):
    # LS projekce reálné části Hs(ω) na span{|ω| cos(ω t_k)} — podpis on-line složky
    pos = np.where(om > 0)[0]
    if t_zeros.size == 0 or pos.size == 0:
        return Hs
    w = om[pos]; y = np.real(Hs[pos])
    B = (np.abs(w)[:, None]) * np.cos(np.outer(w, t_zeros))
    c, *_ = np.linalg.lstsq(B, y, rcond=rcond)
    y_fit = B @ c
    H2 = Hs.copy()
    H2[pos] = (y - y_fit) + 1j*np.imag(H2[pos])
    neg = np.where(om < 0)[0]
    H2[neg] = np.conj(H2[-neg])
    return H2

def lowfreq_notch(Hs, om, window, cycles_cut=6.0, kill_bins=4):
    wcut = 2.0*np.pi * cycles_cut / float(window)
    H2 = Hs.copy()
    H2[np.abs(om) < wcut] = 0.0
    # kill a few lowest bins as a safety belt
    order = np.argsort(np.abs(np.fft.fftfreq(len(H2))))
    killed = 0
    for idx in order:
        if idx == 0: continue
        H2[idx] = 0.0
        killed += 1
        if killed >= 2*kill_bins: break
    return H2

def symmetrize_real_spectrum(Hs):
    N = len(Hs)
    H2 = Hs.copy()
    for k in range(1, N//2):
        H2[-k] = np.conj(H2[k])
    return H2


# ======================== detrend & metrics =========================

def core_indices(N, ignore_frac):
    lo = int(np.floor(ignore_frac * N))
    hi = int(np.ceil((1.0 - ignore_frac) * N))
    return lo, hi

def poly_detrend_in_core(t, h, lo, hi, order=3):
    t0 = t[int(0.5*(lo+hi))]
    x = (t[lo:hi] - t0); y = h[lo:hi]
    V = np.vander(x, N=order+1, increasing=True)
    coeff, *_ = np.linalg.lstsq(V, y, rcond=None)
    X = np.vander(t - t0, N=order+1, increasing=True)
    return h - (X @ coeff)


# ======================== main =========================

if __name__ == "__main__":
    mp.mp.dps = MP_DPS

    # analysis grid
    t, dt = make_t_grid(CENTER_T, WINDOW, POINTS)
    N = len(t)
    lo, hi = core_indices(N, IGNORE_EDGE_FRAC)

    print(f"Running GPN falsification with BACKEND='{BACKEND}' at t~{CENTER_T:,.2f}")
    # zeros detection and Φcrit per backend
    margin = ZERO_MARGIN_FRAC * WINDOW
    t_lo = float(t[0] - margin); t_hi = float(t[-1] + margin)

    if BACKEND == "rs":
        expected = (t_hi - t_lo) / mean_zero_spacing(CENTER_T)
        print(f"Expected zeros in detection range ≈ {expected:.0f}")
        _, t_zeros = detect_zeros_dense(CENTER_T, t_lo, t_hi, ZERO_OVERSAMPLE, backend="rs")
        print(f"Detected {len(t_zeros)} zeros in [{t_lo:.1f}, {t_hi:.1f}]")
        phi_crit = phi_crit_from_backend_RS(t)

    elif BACKEND == "afe":
        expected = (t_hi - t_lo) / mean_zero_spacing(CENTER_T)
        print(f"Expected zeros in detection range ≈ {expected:.0f}")
        _, t_zeros = detect_zeros_dense(CENTER_T, t_lo, t_hi, ZERO_OVERSAMPLE, backend="afe")
        print(f"Detected {len(t_zeros)} zeros in [{t_lo:.1f}, {t_hi:.1f}]")
        phi_crit = phi_crit_from_backend_AFE(t)

    elif BACKEND == "external":
        # load series and zeros provided by OS/Hiary/etc.
        t_ext, y_ext = load_external_series(EXTERNAL_CSV)
        # sanity: enforce same grid and window
        if len(t_ext) != len(t) or abs(t_ext[0]-t[0])>1e-12 or abs(t_ext[-1]-t[-1])>1e-12:
            raise ValueError("External CSV grid must match analysis grid (same t, same spacing).")
        phi_crit = phi_crit_from_external(t_ext, y_ext)
        zeros_all = load_zeros(EXTERNAL_ZEROS)
        mask = (zeros_all >= t_lo) & (zeros_all <= t_hi)
        t_zeros = np.array(zeros_all[mask], dtype=float)
        print(f"Loaded {len(t_zeros)} zeros in [{t_lo:.1f}, {t_hi:.1f}] from file.")
    else:
        raise ValueError("BACKEND must be 'rs', 'afe', or 'external'.")

    # improved background curvature
    B = background_curvature_improved(t, method=B_METHOD, t_switch=B_T_SWITCH, terms=B_TERMS)

    # core signal H_z = d2dt φcrit - B
    Hz = d2dt_spectral(phi_crit, dt) - B

    # loop over Poisson parameters a; project online; LF cleanup; detrend; local metrics
    results = {}
    for a in A_LIST:
        Hs_pad, om_pad, left, right, M = poisson_smooth_fft_padded(
            Hz, dt, a, pad_factor=PAD_FACTOR, taper_alpha=TAPER_ALPHA
        )
        Hs_clean = project_out_online_component(Hs_pad, om_pad, t_zeros, rcond=1e-4)
        Hs_clean = lowfreq_notch(Hs_clean, om_pad, WINDOW*(1+2*ZERO_MARGIN_FRAC),
                                 cycles_cut=LOWFREQ_CYCLES_CUT, kill_bins=KILL_LOW_BINS)
        Hs_clean = symmetrize_real_spectrum(Hs_clean)
        h_pad = np.fft.ifft(Hs_clean).real
        h_off = h_pad[left:left+N]
        h_off = poly_detrend_in_core(t, h_off, lo, hi, order=CORE_POLY_DETREND_ORDER)

        vmax = -np.inf
        for L in L_LIST:
            kern = kappa_gaussian(L, dx=dt)
            h_loc = np.convolve(h_off, kern, mode="same")
            core_max = float(np.max(np.abs(h_loc[lo:hi])))
            vmax = max(vmax, core_max)
            print(f"a={a:g}, L={L:g} | core max = {core_max:.3e}")
        results[a] = vmax

    # summary
    print("\nResiduals in core (max over L):")
    for a in sorted(results.keys(), reverse=True):
        vmax = results[a]
        flag = "OK" if vmax <= EPSILON else "RESIDUAL>eps"
        print(f"  a={a:.6g} | max_core={vmax:.3e}  →  {flag}")
