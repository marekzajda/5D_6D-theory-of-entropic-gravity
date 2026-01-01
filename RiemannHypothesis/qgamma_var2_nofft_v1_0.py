#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qgamma_var2_nofft_v1_0.py

Step 3 (Var2): numerical domination of the archimedean / gamma part Q[g]
under the SAME angular FT convention used in Var2 prime-term control.

We assume an even generator:
    g(x) = (sinc(eps*x))^N * exp(-delta*x^2),
where sinc(z) = sin(z)/z with sinc(0)=1.

We compute:
    Q[g] = ∫ g(x) K_gamma(x) dx + alpha * \hat g(0) + beta * g(0),
with
    \hat g(0) = ∫ g(x) dx,
and K_gamma given (default) by the explicit kernel used in your draft:

    K_gamma(x) =  pi / sinh(pi|x|)  -  1/2 * coth(|x|/2)  -  cosh(|x|/2).

This is a real-even kernel. All computations are real.

We also compute the weighted energy:
    I_W1 = ∫ |g'(x)|^2 * exp(pi|x|) dx.

Then the empirical ratio:
    C1_emp = |Q[g]| / I_W1

Outputs CSV over eps,delta grid.

IMPORTANT:
- If your final explicit formula uses different additive constants,
  set alpha,beta accordingly in CLI.
- If your kernel differs, replace K_gamma() accordingly (single function).
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
    # vectorized sinc(z)=sin(z)/z with sinc(0)=1
    out = np.ones_like(z, dtype=np.float64)
    mask = (z != 0.0)
    out[mask] = np.sin(z[mask]) / z[mask]
    return out

def coth(x):
    # coth(x)=cosh(x)/sinh(x) with series-safe handling
    # For small x: coth(x) ~ 1/x + x/3 - x^3/45 + ...
    ax = np.abs(x)
    out = np.empty_like(x, dtype=np.float64)
    small = ax < 1e-6
    # series for small
    xs = x[small]
    out[small] = (1.0/xs) + xs/3.0 - (xs**3)/45.0
    # normal for others
    xn = x[~small]
    out[~small] = np.cosh(xn) / np.sinh(xn)
    return out

def inv_sinh(x):
    # 1/sinh(x) with series-safe handling
    # For small x: 1/sinh(x) ~ 1/x - x/6 + 7x^3/360 ...
    ax = np.abs(x)
    out = np.empty_like(x, dtype=np.float64)
    small = ax < 1e-6
    xs = x[small]
    out[small] = (1.0/xs) - xs/6.0 + 7.0*(xs**3)/360.0
    xn = x[~small]
    out[~small] = 1.0 / np.sinh(xn)
    return out


# -------------------------
# Gamma kernel (EDIT HERE if needed)
# -------------------------

def K_gamma(x):
    """
    Real-even gamma kernel, vectorized.

    Default kernel from your lemma draft:
        K(x)= π/sinh(π|x|) - 1/2*coth(|x|/2) - cosh(|x|/2)

    Note: uses |x|, so evenness is built-in.
    """
    ax = np.abs(x).astype(np.float64)
    # Avoid exact zero in denominators by relying on series expansions
    term1 = math.pi * inv_sinh(math.pi * ax)              # π / sinh(π|x|)
    term2 = 0.5 * coth(0.5 * ax)                          # (1/2) coth(|x|/2)
    term3 = np.cosh(0.5 * ax)                             # cosh(|x|/2)
    return term1 - term2 - term3


# -------------------------
# Generator g and derivative g'
# -------------------------

def build_grid(L, Nx):
    x = np.linspace(-L, L, Nx, dtype=np.float64)
    dx = float(x[1] - x[0])
    return x, dx

def g_and_gp(x, eps, delta, N):
    """
    g(x) = (sinc(eps x))^N * exp(-delta x^2)
    g'(x) computed analytically with stable handling near 0.
    """
    ex = eps * x
    s = sinc(ex)                 # sin(eps x)/(eps x)
    gauss = np.exp(-delta * x*x)

    g = (s ** N) * gauss

    # derivative of sinc(eps x):
    # d/dx [sin(eps x)/(eps x)] = eps * ( (eps x)cos(eps x) - sin(eps x) ) / (eps x)^2
    #                           = ( (ex)cos(ex) - sin(ex) ) / (x * ex)
    # safer: use series near 0
    ds = np.empty_like(x, dtype=np.float64)
    small = np.abs(ex) < 1e-6
    xs = x[small]
    # sinc(ex) ~ 1 - ex^2/6 + ...
    # derivative w.r.t x: d/dx sinc(ex) = eps * d/d(ex) sinc(ex)
    # d/d(ex) sinc(ex) ~ -(ex)/3 + ...
    ds[small] = eps * (-(eps*xs)/3.0)

    xn = x[~small]
    exn = ex[~small]
    ds[~small] = eps * ((exn * np.cos(exn) - np.sin(exn)) / (exn * exn))

    # g' = N*s^(N-1)*ds * gauss + s^N * gauss' ; gauss' = -2 delta x gauss
    gp = (N * (s ** (N - 1)) * ds) * gauss + (s ** N) * (-2.0 * delta * x) * gauss
    return g, gp


# -------------------------
# Q[g] and energies
# -------------------------

def trapz(y, x):
    return float(np.trapezoid(y, x))

def compute_Q_and_energy(eps, delta, N, Nx, tol_tail, alpha, beta):
    """
    Choose L by Gaussian tail: exp(-delta L^2) ~ tol_tail
    L = sqrt(log(1/tol)/delta).
    """
    L = math.sqrt(max(1.0, math.log(1.0 / tol_tail)) / delta)
    # Give a bit of margin (oscillatory sinc part)
    L *= 1.25

    x, dx = build_grid(L, Nx)
    g, gp = g_and_gp(x, eps, delta, N)

    # Q integral piece
    Kg = K_gamma(x)
    Q_int = trapz(g * Kg, x)

    # hat g(0) = ∫ g(x) dx
    hat0 = trapz(g, x)

    # g(0)
    mid = Nx // 2
    g0 = float(g[mid])

    Q = Q_int + alpha * hat0 + beta * g0

    # energies
    W1 = np.exp(math.pi * np.abs(x))
    IW1 = trapz((gp * gp) * W1, x)

    Iplain = trapz((gp * gp), x)

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
        "Iplain": Iplain,
        "C1_emp": (abs(Q) / IW1) if IW1 > 0 else float("nan"),
        "C0_emp": (abs(Q) / Iplain) if Iplain > 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="0.1,0.3,1.0", type=str)
    ap.add_argument("--delta", default="0.05,0.2,1.0", type=str)
    ap.add_argument("--N", default=4, type=int)
    ap.add_argument("--Nx", default=262144, type=int)
    ap.add_argument("--tol_tail", default=1e-14, type=float)
    ap.add_argument("--alpha", default=1.0, type=float)
    ap.add_argument("--beta", default=0.0, type=float)
    ap.add_argument("--out", default="qgamma_step3_var2.csv", type=str)
    args = ap.parse_args()

    eps_list = parse_list(args.eps)
    deltas = parse_list(args.delta)

    print("Step 3: Q_gamma domination (Var2, no-FFT)")
    print(f"N={args.N}, Nx={args.Nx}, tol_tail={args.tol_tail:g}, alpha={args.alpha}, beta={args.beta}")
    print("Kernel: K(x)= π/sinh(π|x|) - 1/2*coth(|x|/2) - cosh(|x|/2)")
    print("W1(x)=exp(pi|x|)")

    rows = []
    t0 = perf_counter()
    for d in deltas:
        print(f"\n--- delta={d} ---")
        for e in eps_list:
            t1 = perf_counter()
            out = compute_Q_and_energy(
                eps=e, delta=d, N=args.N, Nx=args.Nx,
                tol_tail=args.tol_tail, alpha=args.alpha, beta=args.beta
            )
            dt = perf_counter() - t1
            out["time_s"] = dt
            rows.append(out)
            print(
                f" eps={e:<4}  |Q|={out['Q_abs']:.6e}  IW1={out['IW1']:.6e}  "
                f"C1_emp={out['C1_emp']:.6e}  (L={out['L']:.3f}, dx={out['dx']:.3e}, {dt:.2f}s)"
            )

    # uniform maxima
    max_C1 = max(r["C1_emp"] for r in rows if math.isfinite(r["C1_emp"]))
    max_C0 = max(r["C0_emp"] for r in rows if math.isfinite(r["C0_emp"]))
    rC1 = max(rows, key=lambda r: (r["C1_emp"] if math.isfinite(r["C1_emp"]) else -1))
    rC0 = max(rows, key=lambda r: (r["C0_emp"] if math.isfinite(r["C0_emp"]) else -1))

    print("\n=== UNIFORM SUMMARY ===")
    print(f"max C1_emp = {max_C1:.6e}  at eps,delta=({rC1['eps']},{rC1['delta']})")
    print(f"max C0_emp = {max_C0:.6e}  at eps,delta=({rC0['eps']},{rC0['delta']})")
    print(f"Total time: {perf_counter()-t0:.2f}s")

    # save CSV
    fields = [
        "eps","delta","N","Nx","L","dx",
        "Q","Q_abs","Q_int","hat0","g0",
        "IW1","Iplain","C1_emp","C0_emp","time_s"
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
