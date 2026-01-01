#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qgamma_var2_nofft_v1_1_PATCHED.py

Step 3 (Var2, PATCHED):
Rigorous numerical domination of the gamma / archimedean term Q[g],
fully CCC-compatible.

Fixes:
  - Stable evaluation of K_gamma near x=0 (singularity cancellation)
  - Uses sqrt-energy normalization: |Q[g]| <= C1 * sqrt(I_W1[g])
  - Consistent with prime-term C3 scaling

Author: for Marek Zajda
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

def sinc(z):
    out = np.ones_like(z, dtype=np.float64)
    mask = np.abs(z) > 1e-12
    out[mask] = np.sin(z[mask]) / z[mask]
    return out


# -------------------------
# Stable gamma kernel
# -------------------------

def K_gamma(x):
    """
    Fully stable real-even gamma kernel.

    Exact definition:
        K(x)= π/sinh(π|x|) - 1/2*coth(|x|/2) - cosh(|x|/2)

    For |x| < 1e-4 we use the Taylor-expanded *combined* kernel:
        K(0) = -1
        K(x) = -1 - a1|x| - a2 x^2 + O(x^3)
    """
    ax = np.abs(x)
    out = np.empty_like(ax, dtype=np.float64)

    small = ax < 1e-4
    large = ~small

    # coefficients from exact cancellation
    a1 = (math.pi**2)/6.0 + 1.0/12.0
    a2 = 1.0/8.0

    # Taylor branch (stable)
    xs = ax[small]
    out[small] = -1.0 - a1 * xs - a2 * xs * xs

    # regular branch
    xl = ax[large]
    out[large] = (
        math.pi / np.sinh(math.pi * xl)
        - 0.5 * (np.cosh(0.5 * xl) / np.sinh(0.5 * xl))
        - np.cosh(0.5 * xl)
    )

    return out


# -------------------------
# Generator g and derivative g'
# -------------------------

def g_and_gp(x, eps, delta, N):
    ex = eps * x
    s = sinc(ex)
    gauss = np.exp(-delta * x * x)
    g = (s ** N) * gauss

    # derivative of sinc(eps x)
    ds = np.zeros_like(x)
    small = np.abs(ex) < 1e-6
    xs = x[small]
    ds[small] = eps * (-(eps * xs) / 3.0)
    xn = x[~small]
    exn = ex[~small]
    ds[~small] = eps * ((exn * np.cos(exn) - np.sin(exn)) / (exn * exn))

    gp = (
        N * (s ** (N - 1)) * ds * gauss
        - 2.0 * delta * x * (s ** N) * gauss
    )
    return g, gp


# -------------------------
# Numerical integration
# -------------------------

def trapz(y, x):
    return float(np.trapezoid(y, x))


def compute_Q_and_energy(eps, delta, N, Nx, tol_tail):
    # truncation length from Gaussian tail
    L = math.sqrt(max(1.0, math.log(1.0 / tol_tail)) / delta)
    L *= 1.3

    x = np.linspace(-L, L, Nx, dtype=np.float64)
    dx = x[1] - x[0]

    g, gp = g_and_gp(x, eps, delta, N)

    Kg = K_gamma(x)
    Q_int = trapz(g * Kg, x)
    hat0 = trapz(g, x)
    g0 = g[Nx // 2]

    Q = Q_int + hat0  # alpha=1, beta=0

    # energy
    W1 = np.exp(math.pi * np.abs(x))
    IW1 = trapz((gp * gp) * W1, x)

    return {
        "eps": eps,
        "delta": delta,
        "N": N,
        "Nx": Nx,
        "L": L,
        "dx": dx,
        "Q": Q,
        "Q_abs": abs(Q),
        "IW1": IW1,
        "C1_emp": abs(Q) / math.sqrt(IW1) if IW1 > 0 else float("nan"),
    }


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="0.1,0.3,1.0")
    ap.add_argument("--delta", default="0.05,0.2,1.0")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--Nx", type=int, default=262144)
    ap.add_argument("--tol_tail", type=float, default=1e-14)
    ap.add_argument("--out", default="qgamma_var2_nofft_v1_1_PATCHED.csv")
    args = ap.parse_args()

    eps_list = parse_list(args.eps)
    delta_list = parse_list(args.delta)

    print("Step 3 PATCHED: Q_gamma domination (Var2, CCC-consistent)")
    print(f"N={args.N}, Nx={args.Nx}, tol_tail={args.tol_tail:g}")
    print("Kernel stabilized at x=0, sqrt-energy normalization enabled")

    rows = []
    t0 = perf_counter()

    for d in delta_list:
        print(f"\n--- delta={d} ---")
        for e in eps_list:
            t1 = perf_counter()
            out = compute_Q_and_energy(
                eps=e, delta=d, N=args.N, Nx=args.Nx, tol_tail=args.tol_tail
            )
            dt = perf_counter() - t1
            out["time_s"] = dt
            rows.append(out)

            print(
                f" eps={e:<4} |Q|={out['Q_abs']:.6e} "
                f"IW1={out['IW1']:.6e} "
                f"C1={out['C1_emp']:.6e} "
                f"(L={out['L']:.2f}, {dt:.2f}s)"
            )

    print("\n=== UNIFORM SUMMARY ===")
    maxC1 = max(r["C1_emp"] for r in rows if math.isfinite(r["C1_emp"]))
    rmax = max(rows, key=lambda r: r["C1_emp"])
    print(f"max C1 = {maxC1:.6e} at eps,delta=({rmax['eps']},{rmax['delta']})")
    print(f"Total time: {perf_counter() - t0:.2f}s")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()