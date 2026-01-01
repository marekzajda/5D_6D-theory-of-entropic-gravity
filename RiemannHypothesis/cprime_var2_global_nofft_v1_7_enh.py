#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Var 2 (angular w = k log p), N fixed (default N=4).
RIGOROUS (no-FFT) bound for prime-term using Gaussian majorant:

  |hat g(w)| <= sqrt(pi/delta) * exp(-w^2/(4 delta))
  w = k log p  (angular)

Then:
  |B_prime[g]| <= 2*sqrt(pi/delta) * S(delta),
  S(delta) = sum_p sum_{k>=1} (log p)/p^{k/2} * exp(-(k log p)^2/(4 delta))

We compute:
  S(delta) <= S_{P,K}(delta) + T_k(P,K;delta) + T_p(P;delta),
with explicit k-tail and explicit p-tail in CLOSED FORM using erfc.

Also computes energy integrals:
  IW2 = ∫ |g'(x)|^2 * (x^2 * exp(pi|x|/2)) dx
  IW3 = ∫ |g'(x)|^2 * ( 1_{|x|<=1} + x^2 exp(pi|x|/2) ) dx
for g(x) = sinc(eps x)^N * exp(-delta x^2).

Outputs per (eps,delta):
  Bprime_rig (same for eps, depends only on delta via this majorant),
  IW2, IW3,
  C3W2_rig = Bprime_rig / sqrt(IW2),
  C3W3_rig = Bprime_rig / sqrt(IW3).

Author: for Marek Zajda
"""

import argparse
import math
import time
import csv
from typing import List, Tuple

import numpy as np


# ----------------------------
# primes up to P (byte sieve)
# ----------------------------
def primes_up_to(P: int) -> np.ndarray:
    if P < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(P + 1, dtype=np.uint8)
    sieve[:2] = 0
    r = int(P**0.5)
    for i in range(2, r + 1):
        if sieve[i]:
            sieve[i*i:P+1:i] = 0
    return np.nonzero(sieve)[0].astype(np.int64)


# ----------------------------
# S_{P,K} and tails (rigorous)
# ----------------------------
def S_PK_delta(primes: np.ndarray, P: int, K: int, delta: float) -> float:
    """S_{P,K}(delta) = sum_{p<=P} sum_{1<=k<=K} log p / p^{k/2} * exp(-(k log p)^2/(4 delta))"""
    p = primes  # already <= P
    logp = np.log(p.astype(np.float64))
    out = 0.0
    for k in range(1, K + 1):
        w = k * logp
        out += np.sum(logp * np.exp(-(0.5 * k) * logp) * np.exp(-(w*w) / (4.0 * delta)))
    return float(out)


def T_k_tail(primes: np.ndarray, K: int, delta: float) -> float:
    """
    k-tail:
      sum_{p<=P} log p * p^{-(K+1)/2} * exp(-((K+1)log p)^2/(4 delta)) / (1 - p^{-1/2})
    """
    p = primes.astype(np.float64)
    logp = np.log(p)
    kp = K + 1
    w = kp * logp
    denom = 1.0 - p**(-0.5)
    term = logp * np.exp(-(0.5 * kp) * logp) * np.exp(-(w*w) / (4.0 * delta)) / denom
    return float(np.sum(term))


def T_p_tail_closed_form(P: int, delta: float) -> float:
    """
    p-tail majorant (rigorous):
      T_p(P;delta) <= 1/(1-P^{-1/2}) * ∫_{log P}^∞ u exp(u/2) exp(-u^2/(4 delta)) du
    Closed form via completing the square:
      exponent = -u^2/(4d) + u/2 = d/4 - (u-d)^2/(4d)
      let y = (u-d)/(2 sqrt(d)), y0 = (log P - d)/(2 sqrt(d))
      integral = exp(d/4) * [ d*sqrt(pi*d)*erfc(y0) + 2d*exp(-y0^2) ].
    """
    if P < 2:
        raise ValueError("P must be >= 2")
    u0 = math.log(P)
    sqrt_d = math.sqrt(delta)
    y0 = (u0 - delta) / (2.0 * sqrt_d)
    pref = 1.0 / (1.0 - P**(-0.5))
    val = math.exp(delta / 4.0) * (delta * math.sqrt(math.pi * delta) * math.erfc(y0) + 2.0 * delta * math.exp(-y0*y0))
    return pref * val


def Bprime_rigorous(P: int, K: int, delta: float) -> Tuple[float, float, float, float]:
    """
    Returns:
      Bprime_rig = 2*sqrt(pi/delta) * (S_PK + T_k + T_p)
      and components: S_PK, T_k, T_p
    """
    primes = primes_up_to(P)
    primes = primes[primes <= P]
    S_pk = S_PK_delta(primes, P, K, delta)
    T_k = T_k_tail(primes, K, delta)
    T_p = T_p_tail_closed_form(P, delta)
    S_total = S_pk + T_k + T_p
    Bprime = 2.0 * math.sqrt(math.pi / delta) * S_total
    return Bprime, S_pk, T_k, T_p


# ----------------------------
# Generator g and derivative g'
# ----------------------------
def sinc_stable(ax: np.ndarray) -> np.ndarray:
    """sinc(ax) = sin(ax)/ax, stable near 0"""
    out = np.empty_like(ax, dtype=np.float64)
    small = np.abs(ax) < 1e-6
    # series: sin z / z = 1 - z^2/6 + z^4/120
    z = ax[small]
    out[small] = 1.0 - (z*z)/6.0 + (z**4)/120.0
    out[~small] = np.sin(ax[~small]) / ax[~small]
    return out


def sinc_prime_stable(ax: np.ndarray, eps: float) -> np.ndarray:
    """
    derivative w.r.t x of sinc(eps x) where ax = eps x.
    If s(x)=sin(ax)/ax with ax=eps x, then:
      s'(x) = (ax cos(ax) - sin(ax)) / (ax^2) * eps? careful:
    Using ax=eps x:
      d/dx [sin(ax)/(ax)] = eps * (ax cos(ax) - sin(ax)) / (ax^2).
    """
    out = np.empty_like(ax, dtype=np.float64)
    small = np.abs(ax) < 1e-6
    z = ax[small]
    # series: sinc(z)=1 - z^2/6 + z^4/120
    # ds/dz = -z/3 + z^3/30, and ds/dx = eps * ds/dz
    out[small] = eps * (-(z)/3.0 + (z**3)/30.0)
    out[~small] = eps * ((ax[~small] * np.cos(ax[~small]) - np.sin(ax[~small])) / (ax[~small]**2))
    return out


def g_and_gprime(x: np.ndarray, eps: float, delta: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
    ax = eps * x
    s = sinc_stable(ax)              # sinc(eps x)
    sp = sinc_prime_stable(ax, eps)  # d/dx sinc(eps x)
    gauss = np.exp(-delta * x*x)
    sN = s**N
    g = sN * gauss
    # g' = gauss*( N*s^{N-1}*s' - 2 delta x s^N )
    gprime = gauss * (N * (s**(N-1)) * sp - 2.0 * delta * x * sN)
    return g, gprime


# ----------------------------
# Energy integrals IW2, IW3 (trapz on adaptive grid)
# ----------------------------
def choose_grid(delta: float, eps: float, Nx: int) -> Tuple[np.ndarray, float]:
    """
    Choose symmetric grid [-L, L] adapted to delta and exponential weight exp(pi|x|/2).

    Since integrand has factor exp(-2 delta x^2) * exp(pi x/2) (for x>0),
    the exponent is -2d x^2 + (pi/2) x = -2d (x - pi/(8d))^2 + pi^2/(32 d).
    Peak occurs near x* = pi/(8d). We include that plus a few std dev ~ 1/sqrt(d).
    """
    d = delta
    x_star = math.pi / (8.0 * d)
    # cover peak and tails:
    L = x_star + 10.0 / math.sqrt(d) + 5.0 / max(eps, 1e-6)
    # keep L reasonable lower bound:
    L = max(L, 8.0)
    x = np.linspace(-L, L, Nx, dtype=np.float64)
    return x, L


def compute_IW2_IW3(eps: float, delta: float, N: int, Nx: int) -> Tuple[float, float, float]:
    """
    Returns:
      IW2 = ∫ |g'|^2 * (x^2 exp(pi|x|/2)) dx
      IW3 = ∫ |g'|^2 * ( 1_{|x|<=1} + x^2 exp(pi|x|/2) ) dx
      L used
    """
    x, L = choose_grid(delta, eps, Nx)
    _, gp = g_and_gprime(x, eps, delta, N)
    gp2 = gp*gp

    absx = np.abs(x)
    W2 = (x*x) * np.exp((math.pi/2.0) * absx)
    IW2 = float(np.trapz(gp2 * W2, x))

    W3 = W2.copy()
    W3[absx <= 1.0] += 1.0
    IW3 = float(np.trapz(gp2 * W3, x))

    return IW2, IW3, L


# ----------------------------
# CLI + main
# ----------------------------
def parse_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=200000)
    ap.add_argument("--Kmax", type=int, default=8)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--eps", type=str, default="0.1,0.3,1.0")
    ap.add_argument("--delta", type=str, default="0.05,0.2,1.0")
    ap.add_argument("--Nx", type=int, default=262144, help="grid points for IW2/IW3 (default 262144)")
    ap.add_argument("--out", type=str, default="cprime_var2_global_nofft_v1_7.csv")
    args = ap.parse_args()

    eps_list = parse_list(args.eps)
    delta_list = parse_list(args.delta)

    print(f"Var2 global no-FFT (rigorous prime tails)")
    print(f"P={args.P}, Kmax={args.Kmax}, N={args.N}, Nx={args.Nx}")
    print("FT: hat g(w)=∫ g(x)e^{-iwx} dx, angular w=k log p")
    print("Bound: |hat g(w)| <= sqrt(pi/delta)*exp(-w^2/(4 delta))")
    print()

    rows = []
    for d in delta_list:
        t0 = time.time()
        Bp, S_pk, Tk, Tp = Bprime_rigorous(args.P, args.Kmax, d)
        tBp = time.time() - t0
        print(f"--- delta={d} ---")
        print(f"Bprime_rig = {Bp:.12e}  (S_PK={S_pk:.6e}, Tk={Tk:.6e}, Tp={Tp:.6e})  time={tBp:.2f}s")

        for e in eps_list:
            t1 = time.time()
            IW2, IW3, L = compute_IW2_IW3(e, d, args.N, args.Nx)
            tI = time.time() - t1

            C3W2 = Bp / math.sqrt(IW2) if IW2 > 0 else float("inf")
            C3W3 = Bp / math.sqrt(IW3) if IW3 > 0 else float("inf")

            print(f" eps={e:<4}  IW2={IW2:.6e}  IW3={IW3:.6e}  L={L:.3f}  "
                  f"C3W2_rig={C3W2:.6e}  C3W3_rig={C3W3:.6e}  time_I={tI:.2f}s")

            rows.append({
                "eps": e,
                "delta": d,
                "P": args.P,
                "Kmax": args.Kmax,
                "N": args.N,
                "Bprime_rig": Bp,
                "S_PK": S_pk,
                "T_k": Tk,
                "T_p": Tp,
                "IW2": IW2,
                "IW3": IW3,
                "C3W2_rig": C3W2,
                "C3W3_rig": C3W3,
                "L": L,
                "Nx": args.Nx,
                "time_Bp_s": tBp,
                "time_I_s": tI,
            })
        print()

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
