import numpy as np
import mpmath as mp

# ===================== settings =====================
MP_DPS    = 120          # mpmath precision
T_MIN     = -30.0
T_MAX     =  30.0
N_SAMPLES = 8192

A0        = 1.0          # base Poisson a
J         = 5            # levels: a_j = A0 * 2^{-j}, j=0..J
L0        = 1.0          # base local scale
NL        = 4            # L_n = L0 / n
EPSILON   = 1e-4         # threshold after delta subtraction
IGNORE_EDGE_FRAC = 0.15  # ignore first/last 15% region when taking max

# ================== core math helpers =================
def log_abs_xi(t_grid, prec=MP_DPS):
    mp.mp.dps = prec
    out = []
    half = mp.mpf('0.5')
    for t in t_grid:
        s = half + 1j * mp.mpf(t)
        xi = (half * s * (s - 1)
              * mp.power(mp.pi, -s/2)
              * mp.gamma(s/2)
              * mp.zeta(s))
        out.append(float(mp.log(abs(xi))))
    return np.array(out, dtype=float)

def background_curvature(t_grid, prec=MP_DPS):
    mp.mp.dps = prec
    out = []
    for t in t_grid:
        z = mp.mpf('0.25') + 0.5j * mp.mpf(t)
        psi1 = mp.polygamma(1, z)           # trigamma
        out.append(-0.25 * float(mp.re(psi1)))
    return np.array(out, dtype=float)

def d2dt_spectral(f_vals, dt):
    N = len(f_vals)
    omega = 2 * np.pi * np.fft.fftfreq(N, d=dt)
    f_hat = np.fft.fft(f_vals)
    return np.real(np.fft.ifft(-(omega**2) * f_hat))

def omega_axis(delta_t, N):
    return 2 * np.pi * np.fft.fftfreq(N, d=delta_t)

def poisson_kernel(a, x):
    return a / (np.pi * (a*a + x*x))

def kappa_gaussian(L, dx=1.0):
    # jednoduchý normalizovaný Gauss (bez moment-cancel triků)
    M = max(7, int(6 * L / max(dx, 1e-9)))
    if M % 2 == 0:
        M += 1
    x = np.linspace(-3*L, 3*L, M)
    kern = np.exp(-(x**2) / (2 * L**2))
    kern /= np.sum(kern)
    return kern

# ============ Hardy Z & zeros on the line ============
def theta_RS(t):
    z = mp.mpf('0.25') + 0.5j * mp.mpf(t)
    return float(mp.im(mp.loggamma(z)) - 0.5 * mp.mpf(t) * mp.log(mp.pi))

def hardy_Z(t):
    mp.mp.dps = MP_DPS
    th = theta_RS(t)
    s = 0.5 + 1j * mp.mpf(t)
    z = mp.zeta(s)
    return float(mp.re(mp.e**(1j*th) * z))

def find_zeros_on_line(t_grid, max_refine=30):
    Z_vals = [hardy_Z(float(t)) for t in t_grid]
    zeros = []
    for i in range(len(t_grid)-1):
        a, b = float(t_grid[i]), float(t_grid[i+1])
        Za, Zb = Z_vals[i], Z_vals[i+1]
        if Za == 0.0:
            zeros.append(a)
            continue
        if Za * Zb < 0.0:
            # bisection
            for _ in range(max_refine):
                m = 0.5*(a+b)
                Zm = hardy_Z(m)
                if Za*Zm <= 0:
                    b, Zb = m, Zm
                else:
                    a, Za = m, Zm
            zeros.append(0.5*(a+b))
    return np.array(zeros, dtype=float)

# ============== zero-padded Poisson smoothing ==============
def poisson_smooth_with_padding(signal, dt, a, pad_factor=4):
    N = len(signal)
    M = int(pad_factor * N)
    left = (M - N)//2
    right = M - N - left
    sig_pad = np.pad(signal, (left, right), mode='constant', constant_values=0.0)
    omega = omega_axis(dt, M)
    H = np.fft.fft(sig_pad)
    Hs = np.exp(-a * np.abs(omega)) * H
    h_pad = np.fft.ifft(Hs).real
    return h_pad[left:left+N]

# =================== main pipeline ===================
def compute_pa_h_off_flags(t_grid,
                           a0=A0, J_levels=J,
                           L0_val=L0, NL_levels=NL,
                           epsilon=EPSILON):
    N = len(t_grid)
    dt = float(t_grid[1] - t_grid[0])

    # 1) log|xi| a křivost pozadí
    Phi_crit = log_abs_xi(t_grid)
    B = background_curvature(t_grid)

    # 2) Hrubý H_z = d2dt(Phi_crit) - B
    Hz = d2dt_spectral(Phi_crit, dt) - B

    # 3) on-line nuly t_k
    zeros_on_line = find_zeros_on_line(t_grid)

    flags = []
    lo = int(np.floor(IGNORE_EDGE_FRAC * N))
    hi = int(np.ceil((1.0 - IGNORE_EDGE_FRAC) * N))

    for j in range(J_levels + 1):
        a = a0 * (2 ** (-j))

        # 4a) Poissonovo vyhlazení celé křivosti
        h = poisson_smooth_with_padding(Hz, dt, a, pad_factor=4)

        # 4b) Odečti hladkou stopu delt: 2π * sum_k P_a(t - t_k)
        if zeros_on_line.size > 0:
            Psum = np.zeros_like(h)
            for tk in zeros_on_line:
                Psum += poisson_kernel(a, t_grid - tk)
            h_off = h - 2.0*np.pi*Psum
        else:
            h_off = h

        # 5) Lokální mollifikace a test ve středu intervalu
        for n in range(1, NL_levels + 1):
            Ln = L0_val / n
            kern = kappa_gaussian(Ln, dx=dt)
            h_loc = np.convolve(h_off, kern, mode="same")
            core_max = float(np.max(np.abs(h_loc[lo:hi])))
            if core_max > epsilon:
                flags.append((a, Ln, core_max))

    return flags

# ======================= run =======================
if __name__ == "__main__":
    mp.mp.dps = MP_DPS
    t_grid = np.linspace(T_MIN, T_MAX, N_SAMPLES)

    print("Detecting zeros on the critical line (Hardy Z)...")
    zlist = find_zeros_on_line(t_grid)
    print(f"Found ~{len(zlist)} zeros in [{T_MIN}, {T_MAX}]")

    print("Running Poisson smoothing with delta subtraction...")
    flags = compute_pa_h_off_flags(t_grid,
                                   a0=A0, J_levels=J,
                                   L0_val=L0, NL_levels=NL,
                                   epsilon=EPSILON)

    if flags:
        print("Residuals above threshold (a, L, max_residual; core region only):")
        for (a, L, r) in flags:
            print(f"  a={a:.6g}, L={L:.6g}, residual={r:.3e}")
    else:
        print("No residuals above threshold in core region.")
