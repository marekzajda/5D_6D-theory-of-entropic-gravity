import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Načti souhrnnou tabulku z předchozí analýzy
df = pd.read_csv("OUT_BOOL/ALL_intra_logic_metrics.csv")

# Uprav názvy sloupců pro jistotu
df.columns = [c.strip() for c in df.columns]

# Vytvoř grafy po událostech
events = df["Event"].unique()
fig, axs = plt.subplots(1, len(events), figsize=(5*len(events), 4), sharey=True)

for ax, event in zip(axs, events):
    dfe = df[df["Event"] == event]
    ax.plot(dfe["Block"], dfe["Bit-entropy_Hb"], "-ok", label="Entropy $H_b$")
    ax.plot(dfe["Block"], dfe["Symmetry(B0↔B5,B1↔B4,B2↔B3)"], "-ob", label="Symmetry index")
    ax.set_title(event)
    ax.set_xlabel("Block (1–4)")
    ax.set_xticks([1,2,3,4])
    ax.grid(True, alpha=0.4)
    ax.legend()
axs[0].set_ylabel("Value (0–1)")
plt.suptitle("Ω intra-block entropy and symmetry across events", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("OUT_BOOL/block_entropy_symmetry_summary.png", dpi=200)
plt.show()
