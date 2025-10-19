#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intra-event Boolean analysis of Ω 24-bit frames (4 blocks × 6 bits)
-------------------------------------------------------------------
Vstup: jeden nebo více souborů se symboly ve formátu CSV: "bit_index,bit"
      - řádek "i,1" => bit i = 1
      - řádek "i"   => bit i = 0 (implicitně)
      - soubor může mít méně než 24 řádků -> doplní se nulami, vezme se prvních 24

Pro každý event:
  • rozdělí 24b rámec na 4 bloky × 6 bitů (bloky 1..4)
  • spočte intra-blokové metriky a 6×6 booleovské matice
  • uloží CSV a PNG heatmapy + LaTeX tabulku

Použití:
  python omega_intra_logic.py --out OUT_BOOL \
      GW151226=symbols.bin_151226.csv GW170608=symbols.bin_170608.csv GW200220=symbols.bin_200220.csv

Volby:
  --out <dir>        cílová složka (default OUT_BOOL)
  --latex <file>     LaTeX tabulka metrik (default bool_metrics.tex)
  --prefix <str>     prefix do názvů obrázků/tabulek (volitelné)
"""

import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- I/O: načtení 24 bitů -----------------

def load_bits_csv(path: Path, target_len: int = 24) -> np.ndarray:
    """Načti CSV 'bit_index,bit' -> vrátí přesně target_len bitů (dopočítá 0)."""
    idx, bits = [], []
    with open(path, "r", encoding="utf-8") as f:
        # přeskoč hlavičku, pokud tam je
        first = f.readline()
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) == 1:
                try:
                    i = int(parts[0]); b = 0
                except:
                    continue
            else:
                try:
                    i = int(parts[0])
                except:
                    continue
                try:
                    b = int(parts[1])
                except:
                    b = 0
            idx.append(i); bits.append(b)

    if not idx:
        return np.zeros(target_len, dtype=int)

    order = np.argsort(idx)
    idx = np.array(idx, dtype=int)[order]
    bits = np.array(bits, dtype=int)[order]

    full_idx = np.arange(idx.min(), idx.max()+1, dtype=int)
    pos = {i: b for i, b in zip(idx, bits)}
    full_bits = np.array([pos.get(i, 0) for i in full_idx], dtype=int)

    if len(full_bits) < target_len:
        out = np.zeros(target_len, dtype=int)
        out[:len(full_bits)] = full_bits
        return out
    return full_bits[:target_len]

def to_blocks_6(bits24: np.ndarray) -> np.ndarray:
    """24 -> (4,6)"""
    assert bits24.shape[0] >= 24
    return bits24[:24].reshape(4, 6)

# ----------------- Booleovské pomocné -----------------

def parity(bits6: np.ndarray) -> int:
    return int(np.bitwise_xor.reduce(bits6.astype(int)))

def majority(bits6: np.ndarray) -> int:
    return int(int(np.sum(bits6)) >= 3)

def bit_entropy(bits6: np.ndarray) -> float:
    """Shannonovská entropie pro Bernoulli(p), kde p=podíl jedniček v 6 bitech."""
    p = float(np.mean(bits6))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p*np.log2(p) + (1-p)*np.log2(1-p)))

def symmetry_index(bits6: np.ndarray) -> float:
    """Porovná (B0<->B5, B1<->B4, B2<->B3) -> podíl shod (0..1)."""
    pairs = [(0,5),(1,4),(2,3)]
    eq = [int(bits6[i] == bits6[j]) for i,j in pairs]
    return float(np.mean(eq))

def pairwise_logic_matrix(bits6: np.ndarray, op: str) -> np.ndarray:
    """Vrátí 6×6 matici pro zadanou logickou operaci (AND/OR/XOR/NOR) mezi bity bloku."""
    b = bits6.astype(int)
    A = np.zeros((6,6), dtype=int)
    for i in range(6):
        for j in range(6):
            if op == "AND":
                A[i,j] = int(b[i] & b[j])
            elif op == "OR":
                A[i,j] = int(b[i] | b[j])
            elif op == "XOR":
                A[i,j] = int(b[i] ^ b[j])
            elif op == "NOR":
                A[i,j] = int(1 - (b[i] | b[j]))
            else:
                raise ValueError("Unknown op")
    return A

def equal_pairs(bits6: np.ndarray):
    eq, comp = [], []
    for i in range(6):
        for j in range(i+1,6):
            if bits6[i] == bits6[j]:
                eq.append((i,j))
            if bits6[i] == 1 - bits6[j]:
                comp.append((i,j))
    return eq, comp

# ----------------- Vizualizace -----------------

def plot_heatmap(M: np.ndarray, title: str, out_png: Path):
    plt.figure(figsize=(3.8,3.6))
    plt.imshow(M, vmin=0, vmax=1)
    plt.colorbar(label="value")
    plt.xticks(range(6), [f"B{j}" for j in range(6)])
    plt.yticks(range(6), [f"B{i}" for i in range(6)])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ----------------- Analýza jednoho eventu -----------------

def analyse_event(name: str, csv_path: Path, out_dir: Path, prefix: str = "") -> pd.DataFrame:
    bits24 = load_bits_csv(csv_path, target_len=24)
    blocks = to_blocks_6(bits24)

    rows = []
    for bi in range(4):
        blk = blocks[bi]
        # metriky
        ones = int(np.sum(blk))
        Hb = bit_entropy(blk)
        pari = parity(blk)
        maj = majority(blk)
        sym = symmetry_index(blk)
        eq, comp = equal_pairs(blk)

        # uložit matice
        tag = f"{prefix}{name}_B{bi+1}"
        M_and = pairwise_logic_matrix(blk, "AND")
        M_or  = pairwise_logic_matrix(blk, "OR")
        M_xor = pairwise_logic_matrix(blk, "XOR")
        M_nor = pairwise_logic_matrix(blk, "NOR")

        plot_heatmap(M_and, f"{name} — Block {bi+1} — AND", out_dir/f"{tag}_AND.png")
        plot_heatmap(M_or,  f"{name} — Block {bi+1} — OR",  out_dir/f"{tag}_OR.png")
        plot_heatmap(M_xor, f"{name} — Block {bi+1} — XOR", out_dir/f"{tag}_XOR.png")
        plot_heatmap(M_nor, f"{name} — Block {bi+1} — NOR", out_dir/f"{tag}_NOR.png")

        rows.append({
            "Event": name,
            "Block": bi+1,
            "Bits(6)": "".join(str(int(x)) for x in blk),
            "Sum(1s)": ones,
            "Bit-entropy_Hb": round(Hb, 4),
            "Parity_XOR": pari,
            "Majority(≥3 ones)": maj,
            "Symmetry(B0↔B5,B1↔B4,B2↔B3)": round(sym, 3),
            "Equal_pairs(i,j)": str(eq),
            "Complement_pairs(i,j)": str(comp),
        })

    df = pd.DataFrame(rows)
    # uložit per-event CSV
    df.to_csv(out_dir/f"{prefix}{name}_intra_logic.csv", index=False)
    return df

# ----------------- LaTeX tabulka -----------------

def export_latex_table(all_df: pd.DataFrame, out_tex: Path, caption: str = None, label: str = None):
    cap = caption or "Intra-event boolean metrics for Ω 24-bit frames (four 6-bit blocks per event)."
    lab = label or "tab:omega_intra_bool"
    cols = ["Event","Block","Bits(6)","Sum(1s)","Bit-entropy_Hb","Parity_XOR","Majority(≥3 ones)","Symmetry(B0↔B5,B1↔B4,B2↔B3)"]
    # zkrátit názvy sloupců pro LaTeX
    df = all_df[cols].copy()
    df.rename(columns={
        "Sum(1s)": "Sum~1s",
        "Bit-entropy_Hb": "$H_b$",
        "Parity_XOR": "Parity",
        "Majority(≥3 ones)": "Majority",
        "Symmetry(B0↔B5,B1↔B4,B2↔B3)": "Symmetry"
    }, inplace=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write(df.to_latex(index=False, escape=True))
        f.write(f"\n\\\\caption{{{cap}}}\n\\\\label{{{lab}}}\n\\\\end{{table}}\n")

# ----------------- CLI -----------------

def parse_events(argv_list):
    """
    Očekává položky ve tvaru NAME=path/to/symbols.csv
    Vrací dict {NAME: Path}
    """
    mapping = {}
    for item in argv_list:
        if "=" not in item:
            raise ValueError(f"Event argument musí být NAME=path, dostal jsem: {item}")
        name, path = item.split("=", 1)
        mapping[name.strip()] = Path(path.strip())
    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="+", help="Události ve tvaru NAME=path/to/symbols.csv (např. GW151226=symbols.bin_151226.csv)")
    ap.add_argument("--out", type=Path, default=Path("OUT_BOOL"), help="Výstupní složka")
    ap.add_argument("--latex", type=Path, default=Path("bool_metrics.tex"), help="LaTeX tabulka metrik")
    ap.add_argument("--prefix", type=str, default="", help="Prefix do názvů souborů (volitelné)")
    args = ap.parse_args()

    events = parse_events(args.events)
    args.out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for name, csv_path in events.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"{name}: soubor {csv_path} neexistuje")
        df = analyse_event(name, csv_path, args.out, prefix=args.prefix)
        all_rows.append(df)

    all_df = pd.concat(all_rows, axis=0, ignore_index=True)
    all_df.to_csv(args.out / f"{args.prefix}ALL_intra_logic_metrics.csv", index=False)
    export_latex_table(all_df, args.latex)

    print(f"[OK] Hotovo. Výstupy v: {args.out}")
    print(f"- Souhrnné CSV: {args.out / (args.prefix + 'ALL_intra_logic_metrics.csv')}")
    print(f"- LaTeX tabulka: {args.latex}")
    print(f"- Heatmapy: {args.out}/<EVENT>_B<1..4>_AND/OR/XOR/NOR.png")

if __name__ == "__main__":
    main()
