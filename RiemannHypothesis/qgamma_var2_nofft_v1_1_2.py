#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qgamma_var2_nofft_v1_1.py  (PATCHED)

Step 3 (Var2): numerical domination of the archimedean / gamma part Q[g]
under the SAME angular FT convention used in Var2 prime-term control.

Even generator:
    g(x) = (sinc(eps*x))^N * exp(-delta*x^2),
where sinc(z)=sin(z)/z with sinc(0)=1.

Compute:
    Q[g] = ∫ g(x) K_gamma(x) dx + alpha * hat g(0) + beta * g(0),
    hat g(0) = ∫ g(x) dx.

Kernel (default, from your draft):
    K_gamma(x) =  π/sinh(π|x|)  -  1/2*coth(|x|/2)  -  cosh(|x|/2).

Energy:
    IW1 = ∫ |g'(x)|^2 * exp(pi|x|) dx.

CCC-consistent ratio:
    C1 = |Q[g]| / sqrt(IW1).

Outputs CSV over eps,delta grid.

Author: patched for Marek Zajda
"""

import argparse
import csv
import math
import numpy as np
from time import perf_counter


# -------------------------
# Utilities
# -------------------------

def parse_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def sinc(z: np.ndarray) -> np.ndarray:
    """Vectorized sinc(z)=sin(z)/z with stable series near 0."""
    z = z.astype(np.float64)
    out = np.empty_like(z, dtype=np.float64)
    small = np.abs(z) < 1e-6
    zs = z[small]
    # sin z / z = 1 - z^2/6 + z^4/120
    out[small] = 1.0 - (zs*zs)/6.0 + (zs**4)/120.0
    zn = z[~small]
    out[~small] = np.sin(zn) / zn
    return out

def coth(x: np.ndarray) -> np.ndarray:
    """coth(x)=cosh(x)/sinh(x), stable series for small x."""
    x = x.astype(np.float64)
    ax = np.abs(x)
    out = np.empty_like(x, dtype=np.float64)
    small = ax < 1e-6
    xs = x[small]
    # coth x ~ 1/x + x/3 - x^3/45
    out[small] = (1.0/xs) + xs/3.0 - (xs**3)/45.0
    xn = x[~small]
    out[~small] = np.cosh(xn) / np.sinh(xn)
    return out

def inv_sinh(x: np.ndarray) -> np.ndarray:
    """1/sinh(x), stable series for small x."""
    x = x.astype(np.float64)
    ax = np.abs(x)
    out = np.empty_like(x, dtype=np.float64)
    small = ax < 1e-6
    xs = x[small]
    # 1/sinh x ~ 1/x - x/6 + 7 x^3/360
    out[small] = (1.0/xs) - xs/6.0 + 7.0*(xs**3)/360.0
    xn = x[~small]
    out[~small] = 1.0 / np.sinh(xn)
    return out

def trapez(y: np.ndarray, x: np.ndarray) -> float:
    """Float trapezoidal integral with numpy's modern API."""
    return float(np.trapezoid(y, x))


# -------------------------
# Gamma kernel (stabilized)
# -------------------------

def K_gamma_raw(ax: np.ndarray) -> np.ndarray:
    """
    ax = |x| >= 0. Returns K(ax) = π/sinh(π ax) - 1/2*coth(ax/2) - cosh(ax/2).
    Uses stable inv_sinh & coth, but still has cancellations near 0.
    """
    term1 = math.pi * inv_sinh(math.pi * ax)      # π / sinh(π|x|)
    term2 = 0.5 * coth(0.5 * ax)                  # (1/2) coth(|x|/2)
    term3 = np.cosh(0.5 * ax)                     # cosh(|x|/2)
    return term1 - term2 - term3

def K_gamma(x: np.ndarray) -> np.ndarray:
    """
    Stabilized even kernel.
    For very small |x|, evaluate using symmetric limit via a small offset h:
        K(0) ≈ (K(h)+K(2h))/2
    This avoids catastrophic cancellation / infs from individual pieces.
    """
    ax = np.abs(x).astype(np.float64)
    K = K_gamma_raw(ax)

    # patch around 0
    small = ax < 1e-8
    if np.any(small):
        # choose h relative to scale of grid; 1e-6 is safe in float64
        h = 1e-6
        ax_s = ax[small]
        # keep the "shape" by adding ax_s, but dominated by h anyway
        v1 = K_gamma_raw(ax_s + h)
        v2 = K_gamma_raw(ax_s + 2.0*h)
        K[small] = 0.5*(v1 + v2)

    return K


# -------------------------
# Generator g and derivative g'
# -------------------------

def build_grid(L: float, Nx: int):
    x = np.linspace(-L, L, Nx, dtype=np.float64)
    dx = float(x[1] - x[0])
    return x, dx

def dsinc_epsx_dx(x: np.ndarray, eps: float) -> np.ndarray:
    """
    d/dx sinc(eps x), stable near 0.

    Let z = eps x. sinc(z)=sin z / z.
    d/dz sinc(z) = (z cos z - sin z)/z^2.
    Then d/dx = eps * d/dz.
    Series: d/dz sinc(z) ~ -z/3 + z^3/30.
    """
    z = (eps * x).astype(np.float64)
    out = np.empty_like(z, dtype=np.float64)
    small = np.abs(z) < 1e-6
    zs = z[small]
    out[small] = eps * (-(zs)/3.0 + (zs**3)/30.0)
    zn = z[~small]
    out[~small] = eps * ((zn*np.cos(zn) - np.sin(zn)) / (zn*zn))
    return out

def g_and_gp(x: np.ndarray, eps: float, delta: float, N: int):
    """
    g(x) = (sinc(eps x))^N * exp(-delta x^2)
    g'(x) = exp(-d x^2) * [ N*s^(N-1)*s' - 2 d x * s^N ]
    """
    z = eps * x
    s = sinc(z)
    sp = dsinc_epsx_dx(x, eps)
    gauss = np.exp(-delta * x*x)

    sN = s**N
    g = sN * gauss
    gp = gauss * (N * (s**(N-1)) * sp - 2.0 * delta * x * sN)
    return g, gp


# -------------------------
# Q[g] and energy IW1
# -------------------------

def choose_L(delta: float, eps: float, tol_tail: float) -> float:
    """
    Base from Gaussian tail: exp(-delta L^2) ~ tol_tail  => L ~ sqrt(log(1/tol)/delta).
    Add margin for oscillatory sinc and kernel.
    """
    L0 = math.sqrt(max(1.0, math.log(1.0 / tol_tail)) / delta)
    # margins:
    L = 1.35 * L0 + 8.0 + 2.0 / max(eps, 1e-6)
    return float(L)

def compute_Q_and_IW1(eps: float, delta: float, N: int, Nx: int,
                     tol_tail: float, alpha: float, beta: float):
    L = choose_L(delta, eps, tol_tail)
    x, dx = build_grid(L, Nx)

    g, gp = g_and_gp(x, eps, delta, N)

    # Q integral
    Kg = K_gamma(x)
    Q_int = trapez(g * Kg, x)

    # hat g(0) = ∫ g
    hat0 = trapez(g, x)

    # g(0): center index
    mid = Nx // 2
    g0 = float(g[mid])

    Q = Q_int + alpha * hat0 + beta * g0

    # weighted energy IW1 = ∫ |g'|^2 e^{pi|x|}
    W1 = np.exp(math.pi * np.abs(x))
    IW1 = trapez((gp*gp) * W1, x)

    return {
        "eps": eps,
        "delta": delta,
        "N": N,
        "Nx": Nx,
        "L": L,
        "dx": dx,
        "Q": Q,
        "Q_abs": abs(Q),
        "Q_int": Q_int,
        "hat0": hat0,
        "g0": g0,
        "IW1": IW1,
        # CCC-consistent normalization:
        "C1_emp": (abs(Q) / math.sqrt(IW1)) if IW1 > 0 else float("nan"),
    }


# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="0.1,0.3,1.0", type=str)
    ap.add_argument("--delta", default="0.05,0.2,1.0", type=str)
    ap.add_argument("--N", default=4, type=int)
    ap.add_argument("--Nx", default=262144, type=int)
    ap.add_argument("--tol_tail", default=1e-14, type=float)
    ap.add_argument("--alpha", default=1.0, type=float)
    ap.add_argument("--beta", default=0.0, type=float)
    ap.add_argument("--out", default="qgamma_var2_nofft_v1_1.csv", type=str)
    args = ap.parse_args()

    eps_list = parse_list(args.eps)
    deltas = parse_list(args.delta)

    print("Step 3 PATCHED: Q_gamma domination (Var2, CCC-consistent)")
    print(f"N={args.N}, Nx={args.Nx}, tol_tail={args.tol_tail:g}")
    print("Kernel stabilized at x=0, sqrt-energy normalization enabled")
    print()

    rows = []
    t0 = perf_counter()

    for d in deltas:
        print(f"--- delta={d} ---")
        for e in eps_list:
            t1 = perf_counter()
            r = compute_Q_and_IW1(
                eps=e, delta=d, N=args.N, Nx=args.Nx,
                tol_tail=args.tol_tail, alpha=args.alpha, beta=args.beta
            )
            dt = perf_counter() - t1
            r["time_s"] = dt
            rows.append(r)
            print(f" eps={e:<4}  |Q|={r['Q_abs']:.6e} IW1={r['IW1']:.6e} "
                  f"C1={r['C1_emp']:.6e} (L={r['L']:.2f}, {dt:.2f}s)")
        print()

    # uniform summary
    finite = [r for r in rows if math.isfinite(r["C1_emp"])]
    if finite:
        rmax = max(finite, key=lambda rr: rr["C1_emp"])
        print("=== UNIFORM SUMMARY ===")
        print(f"max C1 = {rmax['C1_emp']:.6e} at eps,delta=({rmax['eps']},{rmax['delta']})")
    print(f"Total time: {perf_counter()-t0:.2f}s\n")

    # save CSV
    fields = ["eps","delta","N","Nx","L","dx","Q","Q_abs","Q_int","hat0","g0","IW1","C1_emp","time_s"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
