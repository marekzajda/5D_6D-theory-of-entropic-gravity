# ultra_high_rs_test_final_v4_enhanced.py
# Purpose: High-T GPN falsification pipeline with Riemann–Siegel core
# Code in English; Czech comments explain the steps and design choices.

import numpy as np
import mpmath as mp

# ===================== user parameters =====================
# (Základní nastavení pro běh; uprav podle potřeby.)
MP_DPS    = 120            # přesnost mpmath (vyšší = pomalejší)
CENTER_T  = 1.0e8          # centrum okna (např. 1e8, 1e9, 1e12*)
WINDOW    = 2000.0         # šířka analyzovaného okna v t
POINTS    = 2048           # body v čase (víc = jemnější derivace)
ZERO_OVERSAMPLE  = 8       # zhuštění mřížky pro detekci nul (~6-10)
ZERO_MARGIN_FRAC = 0.40    # přesah pro detekci nul mimo okno (kvůli ocasům)

A0        = 3.0            # základní Poissonovo a
J_LEVELS  = 4              # a_j = A0 * 2^{-j} pro j=0..J_LEVELS
L_LIST    = [1.0, 0.5, 1/3, 0.25]  # lokální Gaussova mollifikace
EPSILON   = 1e-4           # práh pro "core-max" (informační)

IGNORE_EDGE_FRAC = 0.30    # ignoruj 30 % okraje při měření
PAD_FACTOR       = 8       # zero-padding pro FFT (lepší spektrální rozlišení)
LOWFREQ_CYCLES_CUT = 6.0   # zářez: vyřízni |ω| < 2π*cycles/WINDOW
KILL_LOW_BINS    = 4       # po notchi zabij ještě pár nejnižších binů
CORE_POLY_DETREND_ORDER = 3  # detrend 0..3 v jádru

# Poznámka: * T ~ 1e12 je s čistým RS v Pythonu na hraně. Pro 1e15 je vhodný externí backend.


# ===================== helpers: grids, kernels =====================

def make_t_grid(center, window, points):
    # Ekvidistantní mřížka v čase pro analýzu
    t0 = center - 0.5*window
    t1 = center + 0.5*window
    t = np.linspace(t0, t1, int(points))
    return t, float(t[1] - t[0])

def omega_axis(dt, N):
    # Frekvenční osa v radiánech za jednotku t (FFT konvence)
    return 2*np.pi*np.fft.fftfreq(N, d=dt)

def kappa_gaussian(L, dx=1.0):
    # Jednoduchý normalizovaný Gauss jako lokální mollifier
    M = max(7, int(6 * L / max(dx, 1e-12)))
    if M % 2 == 0:
        M += 1
    x = np.linspace(-3*L, 3*L, M)
    w = np.exp(-(x**2)/(2*L**2))
    w /= np.sum(w)
    return w

def d2dt_spectral(f_vals, dt):
    # Druhá derivace v čase přes spektrální filtr (stabilní a přesné)
    N = len(f_vals)
    omega = omega_axis(dt, N)
    f_hat = np.fft.fft(f_vals)
    return np.real(np.fft.ifft(-(omega**2) * f_hat))


# ===================== Riemann–Siegel core =====================

def theta_RS(t):
    # Riemann–Siegelova fáze (spolehlivá i pro velká t)
    z = mp.mpf('0.25') + 0.5j*mp.mpf(t)
    return float(mp.im(mp.loggamma(z)) - 0.5*mp.mpf(t)*mp.log(mp.pi))

def precompute_RS(center_t):
    # Předpočty pro RS: N ~ sqrt(T/(2π)), plus logn a váhy n^{-1/2}
    N = int(np.floor(np.sqrt(float(center_t)/(2.0*np.pi))))
    if N < 50:
        raise ValueError("RS cutoff N too small; increase CENTER_T.")
    n = np.arange(1, N+1, dtype=np.float64)
    return N, np.log(n), n**(-0.5)

def Z_RS(t_vec, N, logn, w, block=4000):
    # Hardyho Z(t) ≈ 2 Σ_{n≤N} n^{-1/2} cos(t log n - θ(t)) – blokově kvůli paměti
    theta = np.array([theta_RS(float(t)) for t in t_vec], dtype=float)
    Z = np.zeros_like(theta)
    m = len(logn)
    for start in range(0, m, block):
        end = min(start+block, m)
        ph = np.outer(t_vec, logn[start:end]) - theta[:, None]
        Z += 2.0 * (np.cos(ph) @ w[start:end])
    return Z

def mean_zero_spacing(T):
    # Střední rozestup nul: Δt ≈ 2π / log(T/(2π))
    return 2.0*np.pi / np.log(T/(2.0*np.pi))


# ===================== zeros on the line (dense + refine) =====================

def detect_zeros_RS_dense(t_lo, t_hi, center_T, oversample=8, refine=True, refine_steps=14):
    # Hustá mřížka ~oversample× jemnější než střední rozestup; zjemnění bisekcí s RS
    N, logn, w = precompute_RS(center_T)
    spacing = float(mean_zero_spacing(center_T))
    dtz = spacing / max(oversample, 2)
    t_grid = np.arange(t_lo, t_hi + 0.5*dtz, dtz, dtype=float)
    Zv = Z_RS(t_grid, N, logn, w)

    idx = np.where(np.sign(Zv[:-1])*np.sign(Zv[1:]) < 0)[0]
    seeds = np.stack([t_grid[idx], t_grid[idx+1]], axis=1)
    if not refine or len(seeds) == 0:
        return 0.5*(seeds[:,0] + seeds[:,1])

    zeros = []
    for a, b in seeds:
        aa, bb = float(a), float(b)
        Za = Z_RS(np.array([aa]), N, logn, w)[0]
        Zb = Z_RS(np.array([bb]), N, logn, w)[0]
        for _ in range(refine_steps):
            m = 0.5*(aa+bb)
            Zm = Z_RS(np.array([m]), N, logn, w)[0]
            if Za*Zm <= 0.0:
                bb, Zb = m, Zm
            else:
                aa, Za = m, Zm
        zeros.append(0.5*(aa+bb))
    return np.array(zeros, dtype=float)


# ===================== Phi_crit, B(t), Hz via RS =====================

def log_abs_xi_via_RS(t_vec):
    # Φ_crit(t) = log|xi(1/2+it)| = log|zeta| + Re(logΓ(s/2) - (s/2)logπ) + log|s(s-1)/2|
    mp.mp.dps = MP_DPS
    N, logn, w = precompute_RS(0.5*(float(t_vec[0])+float(t_vec[-1])))
    Z = Z_RS(t_vec, N, logn, w)
    logabsz = np.log(np.maximum(np.abs(Z), 1e-300))
    phi = np.array(logabsz, dtype=float)
    half = mp.mpf('0.5')
    for i, t in enumerate(t_vec):
        s = half + 1j*mp.mpf(t)
        bg = mp.loggamma(s/2.0) - (s/2.0)*mp.log(mp.pi)
        phi[i] += float(mp.re(bg))
        ss = s*(s-1.0)/2.0
        phi[i] += float(mp.log(abs(ss)))
    return phi

def background_curvature(t_vec):
    # B(t) = -1/4 * Re trigamma(1/4 + i t/2)
    mp.mp.dps = MP_DPS
    out = []
    for t in t_vec:
        z = mp.mpf('0.25') + 0.5j * mp.mpf(t)
        psi1 = mp.polygamma(1, z)  # trigamma
        out.append(-0.25 * float(mp.re(psi1)))
    return np.array(out, dtype=float)


# ===================== Poisson + spectral ops =====================

def poisson_smooth_fft_padded(signal, dt, a, pad_factor=8):
    # Poisson ve frekvenci s center-paddingem (menší aliasing, jemnější ω-mřížka)
    N = len(signal)
    M = int(pad_factor * N)
    left = (M - N)//2
    right = M - N - left
    sig_pad = np.pad(signal, (left, right), mode='constant', constant_values=0.0)
    omega = omega_axis(dt, M)
    H = np.fft.fft(sig_pad)
    Hs = np.exp(-a*np.abs(omega)) * H
    return Hs, omega, left, right, M

def project_out_online_component(Hs, omega, t_zeros):
    # LS projekce reálné části Hs(ω) na span{|ω| cos(ω t_k)} – tvar on-line stopy po Poissonu
    pos = np.where(omega > 0)[0]
    if t_zeros.size == 0 or pos.size == 0:
        return Hs
    w = omega[pos]
    y = np.real(Hs[pos])
    B = (np.abs(w)[:, None]) * np.cos(np.outer(w, t_zeros))
    c, *_ = np.linalg.lstsq(B, y, rcond=None)  # QR/SVD by bylo hezčí; pro praxi OK
    y_fit = B @ c
    Hs_res = Hs.copy()
    Hs_res[pos] = (y - y_fit) + 1j*np.imag(Hs_res[pos])
    neg = np.where(omega < 0)[0]
    Hs_res[neg] = np.conj(Hs_res[-neg])  # zrcadlení kvůli reálnému signálu
    return Hs_res

def lowfreq_notch(Hs, omega, window, cycles_cut=6.0):
    # Tvrdý zářez |ω| < 2π*cycles_cut/WINDOW (odstraní DC a ultranízké frekvence)
    wcut = 2.0*np.pi * cycles_cut / float(window)
    H2 = Hs.copy()
    H2[np.abs(omega) < wcut] = 0.0
    return H2

def kill_low_bins(Hs, k=4):
    # Pojistka: vynuluj několik nejnižších binů podle |freq| (k kladných + k záporných)
    H2 = Hs.copy()
    N = len(H2)
    order = np.argsort(np.abs(np.fft.fftfreq(N)))
    killed = 0
    for idx in order:
        if idx == 0:  # DC přeskoč
            continue
        H2[idx] = 0.0
        killed += 1
        if killed >= 2*k:
            break
    return H2

def symmetrize_real_signal_spectrum(Hs):
    # Zajisti přesnou hermitovskou symetrii (pro reálný časový signál po iFFT)
    N = len(Hs)
    H2 = Hs.copy()
    for k in range(1, N//2):
        H2[-k] = np.conj(H2[k])
    return H2


# ===================== detrend & metrics =====================

def poly_detrend_in_core(t, h, lo, hi, order=3):
    # Odečti polynom 0..order fitovaný JEN v jádru a aplikuj přes celý interval
    t0 = t[int(0.5*(lo+hi))]
    x_core = (t[lo:hi] - t0)
    y_core = h[lo:hi]
    V = np.vander(x_core, N=order+1, increasing=True)
    coeff, *_ = np.linalg.lstsq(V, y_core, rcond=None)
    X = np.vander(t - t0, N=order+1, increasing=True)
    return h - (X @ coeff)

def core_indices(N, ignore_frac):
    lo = int(np.floor(ignore_frac * N))
    hi = int(np.ceil((1.0 - ignore_frac) * N))
    return lo, hi


# ===================== main pipeline =====================

if __name__ == "__main__":
    mp.mp.dps = MP_DPS

    # 1) analyzovaná mřížka a jádro (bez okrajů)
    t, dt = make_t_grid(CENTER_T, WINDOW, POINTS)
    N = len(t)
    lo, hi = core_indices(N, IGNORE_EDGE_FRAC)

    # 2) hustá detekce nul v širším pásmu (kvůli okrajům)
    margin = ZERO_MARGIN_FRAC * WINDOW
    t_lo = float(t[0] - margin)
    t_hi = float(t[-1] + margin)
    spacing = float(mean_zero_spacing(CENTER_T))
    expected = (t_hi - t_lo) / spacing
    print(f"Running Poisson smoothing test at t ~ {CENTER_T:,.2f} ...")
    print(f"Expected zeros in detection range ≈ {expected:.0f}")

    t_zeros = detect_zeros_RS_dense(
        t_lo, t_hi, CENTER_T,
        oversample=ZERO_OVERSAMPLE,
        refine=True, refine_steps=14
    )
    print(f"Detected {len(t_zeros)} zeros in [{t_lo:.1f}, {t_hi:.1f}]")

    # 3) Φ_crit(t), B(t), a hrubé H_z = d2dt Φ_crit - B
    Phi_crit = log_abs_xi_via_RS(t)
    B = background_curvature(t)
    Hz = d2dt_spectral(Phi_crit, dt) - B

    # 4) smyčka přes Poissonova a_j, projekce on-line složky, LF-cleanup, detrend, lokální metriky
    flags = []
    for j in range(J_LEVELS + 1):
        a = A0 * (2 ** (-j))

        # Poisson ve frekvenci + padding (menší aliasing)
        Hs_pad, omega_pad, left, right, M = poisson_smooth_fft_padded(Hz, dt, a, pad_factor=PAD_FACTOR)

        # Projekce on-line složky (kritická přímka) ve frekvenci
        Hs_clean = project_out_online_component(Hs_pad, omega_pad, t_zeros)

        # Low-frequency notch + kill pár nejnižších binů (DC/ultra-low leak)
        Hs_clean = lowfreq_notch(Hs_clean, omega_pad, WINDOW*(1+2*ZERO_MARGIN_FRAC),
                                 cycles_cut=LOWFREQ_CYCLES_CUT)
        Hs_clean = kill_low_bins(Hs_clean, k=KILL_LOW_BINS)

        # Hermitovská symetrie pro přesně reálný časový signál
        Hs_clean = symmetrize_real_signal_spectrum(Hs_clean)

        # Návrat do času a odříznutí padd.
        h_pad = np.fft.ifft(Hs_clean).real
        h_off = h_pad[left:left+N]

        # Detrend v jádru (polynom až 3. řádu) a aplikace na celý interval
        h_off = poly_detrend_in_core(t, h_off, lo, hi, order=CORE_POLY_DETREND_ORDER)

        # Lokální Gaussova mollifikace a měření v jádru
        for L in L_LIST:
            kern = kappa_gaussian(L, dx=dt)
            h_loc = np.convolve(h_off, kern, mode="same")
            core_max = float(np.max(np.abs(h_loc[lo:hi])))
            print(f"a={a:g}, L={L:g} | core max = {core_max:.3e}")
            if core_max > EPSILON:
                flags.append((a, L, core_max))

    # 5) shrnutí (max přes L pro každé a)
    if flags:
        by_a = {}
        for (a, L, r) in flags:
            by_a.setdefault(a, []).append(r)
        print("\nResiduals above threshold in core (a, max_over_L core_max):")
        for a in sorted(by_a.keys(), reverse=True):
            vmax = max(by_a[a])
            print(f"  a={a:.6g}, max_core_max={vmax:.3e}")
    else:
        print("No residuals above threshold in core.")

# OUTPUT:
# C:\Users\Marek Zajda\Desktop\UEST teorie\QUEST 2.0\Riemann hypothesis all>python ultra_high_rs_test_v4_enhanced.py
Running Poisson smoothing test at t ~ 1,000,000,000.00 ...
Expected zeros in detection range ≈ 10821
Detected 10815 zeros in [999998200.0, 1000001800.0]
a=3, L=1 | core max = 2.137e-01
a=3, L=0.5 | core max = 2.137e-01
a=3, L=0.333333 | core max = 2.137e-01
a=3, L=0.25 | core max = 2.137e-01
a=1.5, L=1 | core max = 3.380e-01
a=1.5, L=0.5 | core max = 3.380e-01
a=1.5, L=0.333333 | core max = 3.380e-01
a=1.5, L=0.25 | core max = 3.380e-01
a=0.75, L=1 | core max = 8.556e-01
a=0.75, L=0.5 | core max = 8.556e-01
a=0.75, L=0.333333 | core max = 8.556e-01
a=0.75, L=0.25 | core max = 8.556e-01
a=0.375, L=1 | core max = 1.418e+00
a=0.375, L=0.5 | core max = 1.418e+00
a=0.375, L=0.333333 | core max = 1.418e+00
a=0.375, L=0.25 | core max = 1.418e+00
a=0.1875, L=1 | core max = 1.851e+00
a=0.1875, L=0.5 | core max = 1.851e+00
a=0.1875, L=0.333333 | core max = 1.851e+00
a=0.1875, L=0.25 | core max = 1.851e+00

# Residuals above threshold in core (a, max_over_L core_max):
  a=3, max_core_max=2.137e-01
  a=1.5, max_core_max=3.380e-01
  a=0.75, max_core_max=8.556e-01
  a=0.375, max_core_max=1.418e+00
  a=0.1875, max_core_max=1.851e+00
