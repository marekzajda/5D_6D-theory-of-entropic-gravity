# ultra_high_rs_test_v6.py
# Purpose: High-T GPN falsification pipeline with stronger low-frequency hygiene.
# Code in English; comments in Czech.

import numpy as np
import mpmath as mp

# ===================== user parameters =====================
MP_DPS    = 120            # přesnost mpmath (vyšší = pomalejší)
CENTER_T  = 1.0e9          # centrum okna (např. 1e8, 1e9, 1e12*)
WINDOW    = 2000.0         # šířka analyzovaného okna v t
POINTS    = 4096           # víc bodů -> jemnější derivace a FFT mřížka

ZERO_OVERSAMPLE  = 8       # zhuštění mřížky pro detekci nul (~6-10)
ZERO_MARGIN_FRAC = 0.50    # širší přesah pro nuly (kvůli ocasům)

A_LIST    = [3.0, 2.0, 1.0, 0.5]  # Poissonovy parametry (můžeš rozšířit)
L_LIST    = [1.0, 0.5, 1/3, 0.25] # lokální Gaussova mollifikace
EPSILON   = 1e-4           # práh pro "core-max" (informační)

IGNORE_EDGE_FRAC     = 0.35   # ignoruj 35 % okraje při měření
PAD_FACTOR           = 12     # zero-padding pro FFT (lepší spektrální rozlišení)
LOWFREQ_CYCLES_CUT   = 10.0   # zářez: vyřízni |ω| < 2π*cycles/WINDOW
KILL_LOW_BINS        = 6      # po notchi zlikviduj pár nejnižších binů
CORE_POLY_DETREND_ORDER = 4   # detrend až 4. řádu v jádru
PROJ_RCOND           = 1e-4   # truncation pro lstsq (projekce ve frekvenci)

TAPER_ALPHA          = 0.25   # Tukey apodizace (0=bez, 1=Hann). Snižuje okrajové úniky.

# Pozn.: * T ~ 1e12 je v čistém Python RS pomalé; pro 1e15 použij externí backend.


# ===================== helpers: grids, kernels =====================

def make_t_grid(center, window, points):
    t0 = center - 0.5*window
    t1 = center + 0.5*window
    t = np.linspace(t0, t1, int(points))
    return t, float(t[1] - t[0])

def omega_axis(dt, N):
    return 2*np.pi*np.fft.fftfreq(N, d=dt)

def kappa_gaussian(L, dx=1.0):
    M = max(7, int(6 * L / max(dx, 1e-12)))
    if M % 2 == 0:
        M += 1
    x = np.linspace(-3*L, 3*L, M)
    w = np.exp(-(x**2)/(2*L**2))
    w /= np.sum(w)
    return w

def d2dt_spectral(f_vals, dt):
    N = len(f_vals)
    omega = omega_axis(dt, N)
    f_hat = np.fft.fft(f_vals)
    return np.real(np.fft.ifft(-(omega**2) * f_hat))

def tukey_window(N, alpha=0.25):
    # Apodizace: potlačí okrajové úniky bez zničení jádra.
    if alpha <= 0:
        return np.ones(N)
    if alpha >= 1:
        return np.hanning(N)
    w = np.ones(N)
    L = int(np.floor(alpha*(N-1)/2.0))
    if L <= 0:
        return w
    x = np.linspace(0, 1, L, endpoint=False)
    left  = 0.5*(1 + np.cos(np.pi*(2*x/alpha - 1)))
    right = left[::-1]
    w[:L]  = left
    w[-L:] = right
    return w


# ===================== Riemann–Siegel core =====================

def theta_RS(t):
    # Riemann–Siegelova fáze
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


# ===================== zeros on the line (dense + refine) =====================

def detect_zeros_RS_dense(t_lo, t_hi, center_T, oversample=8, refine=True, refine_steps=14):
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
        psi1 = mp.polygamma(1, z)
        out.append(-0.25 * float(mp.re(psi1)))
    return np.array(out, dtype=float)


# ===================== Poisson + spectral ops =====================

def poisson_smooth_fft_padded(signal, dt, a, pad_factor=8, taper_alpha=0.0):
    # Poisson ve frekvenci s center-paddingem + volitelnou apodizací (Tukey) v čase
    N = len(signal)
    if taper_alpha > 0:
        tap = tukey_window(N, alpha=taper_alpha)  # apodizace proti okrajům
        signal = signal * tap

    M = int(pad_factor * N)
    left = (M - N)//2
    right = M - N - left
    sig_pad = np.pad(signal, (left, right), mode='constant', constant_values=0.0)

    omega = omega_axis(dt, M)
    H = np.fft.fft(sig_pad)
    Hs = np.exp(-a*np.abs(omega)) * H
    return Hs, omega, left, right, M

def project_out_online_component(Hs, omega, t_zeros, rcond=1e-4):
    # LS projekce reálné části Hs(ω) na span{|ω| cos(ω t_k)} – s truncation rcond (stabilizace)
    pos = np.where(omega > 0)[0]
    if t_zeros.size == 0 or pos.size == 0:
        return Hs
    w = omega[pos]
    y = np.real(Hs[pos])
    B = (np.abs(w)[:, None]) * np.cos(np.outer(w, t_zeros))
    c, *_ = np.linalg.lstsq(B, y, rcond=rcond)
    y_fit = B @ c
    Hs_res = Hs.copy()
    Hs_res[pos] = (y - y_fit) + 1j*np.imag(Hs_res[pos])
    neg = np.where(omega < 0)[0]
    Hs_res[neg] = np.conj(Hs_res[-neg])
    return Hs_res

def lowfreq_notch(Hs, omega, window, cycles_cut=6.0, kill_bins=4):
    # Tvrdý zářez |ω| < 2π*cycles_cut/WINDOW + zabití k nejnižších binů
    wcut = 2.0*np.pi * cycles_cut / float(window)
    H2 = Hs.copy()
    mask = np.abs(omega) < wcut
    H2[mask] = 0.0
    # pojistka: zabij pár nejnižších binů podle |freq|
    order = np.argsort(np.abs(np.fft.fftfreq(len(H2))))
    killed = 0
    for idx in order:
        if idx == 0: continue
        H2[idx] = 0.0
        killed += 1
        if killed >= 2*kill_bins:
            break
    return H2

def symmetrize_real_signal_spectrum(Hs):
    # Hermitovská symetrie (reálný časový signál po iFFT)
    N = len(Hs)
    H2 = Hs.copy()
    for k in range(1, N//2):
        H2[-k] = np.conj(H2[k])
    return H2


# ===================== detrend & metrics =====================

def poly_detrend_in_core(t, h, lo, hi, order=3):
    # Odečti polynom 0..order fitovaný v jádru; aplikuj přes celý interval
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

def spectral_energies(Hs_clean, omega):
    # Užitečné do článku: L2 a H^{1/2} energie po projekci (diskrétně)
    domega = abs(omega[1] - omega[0])
    E_L2  = (domega / (2*np.pi)) * np.sum(np.abs(Hs_clean)**2)
    E_H12 = (domega / (2*np.pi)) * np.sum(np.abs(omega) * np.abs(Hs_clean)**2)
    return float(E_L2), float(E_H12)


# ===================== main pipeline =====================

if __name__ == "__main__":
    mp.mp.dps = MP_DPS

    # 1) analyzovaná mřížka a jádro
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

    # 3) Φ_crit(t), B(t), Hrubé H_z = d2dt Φ_crit - B
    Phi_crit = log_abs_xi_via_RS(t)
    B = background_curvature(t)
    Hz = d2dt_spectral(Phi_crit, dt) - B

    # 4) smyčka přes a: Poisson+padding(+taper), projekce, LF notch, detrend, lokální metriky
    results = {}
    for a in A_LIST:
        # Poisson ve frekvenci, s paddingem a apodizací v čase
        Hs_pad, omega_pad, left, right, M = poisson_smooth_fft_padded(
            Hz, dt, a, pad_factor=PAD_FACTOR, taper_alpha=TAPER_ALPHA
        )

        # Projekce on-line složky (kritická přímka) s rcond stabilizací
        Hs_clean = project_out_online_component(Hs_pad, omega_pad, t_zeros, rcond=PROJ_RCOND)

        # Low-frequency notch + kill pár nejnižších binů
        Hs_clean = lowfreq_notch(
            Hs_clean, omega_pad, WINDOW*(1+2*ZERO_MARGIN_FRAC),
            cycles_cut=LOWFREQ_CYCLES_CUT, kill_bins=KILL_LOW_BINS
        )

        # Hermitovská symetrie
        Hs_clean = symmetrize_real_signal_spectrum(Hs_clean)

        # zpět do času + odříznutí padd.
        h_pad = np.fft.ifft(Hs_clean).real
        h_off = h_pad[left:left+N]

        # detrend v jádru (polynom 0..CORE_POLY_DETREND_ORDER)
        h_off = poly_detrend_in_core(t, h_off, lo, hi, order=CORE_POLY_DETREND_ORDER)

        # energie po projekci (užitečné do článku)
        E_L2, E_H12 = spectral_energies(Hs_clean, omega_pad)

        # lokální Gaussova mollifikace a měření v jádru
        vmax = -np.inf
        for L in L_LIST:
            kern = kappa_gaussian(L, dx=dt)
            h_loc = np.convolve(h_off, kern, mode="same")
            core_max = float(np.max(np.abs(h_loc[lo:hi])))
            vmax = max(vmax, core_max)
            print(f"a={a:g}, L={L:g} | core max = {core_max:.3e}, E_L2={E_L2:.3e}, E_H12={E_H12:.3e}")
        results[a] = (vmax, E_L2, E_H12)

    # 5) shrnutí
    print("\nResiduals in core (max over L) and energies after projection:")
    for a in sorted(results.keys(), reverse=True):
        vmax, E_L2, E_H12 = results[a]
        flag = "OK" if vmax <= EPSILON else "RESIDUAL>eps"
        print(f"  a={a:.6g} | max_core={vmax:.3e} ({flag}),  E_L2={E_L2:.3e},  E_H12={E_H12:.3e}")

