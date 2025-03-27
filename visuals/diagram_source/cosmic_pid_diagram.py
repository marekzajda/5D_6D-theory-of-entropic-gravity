import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow

# Vytvoření figurky s vysokým rozlišením
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# 5D prostor (modrý kruh)
bulk = Circle((0.5, 0.5), 0.3, fill=False, 
              edgecolor='blue', linewidth=3, label='5D Bulk')
ax.add_patch(bulk)

# Holografická plocha (průhledná)
screen = Circle((0.5, 0.5), 0.2, fill=True, 
                color='blue', alpha=0.1, label='Holografická plocha')
ax.add_patch(screen)

# Šipky
ax.arrow(0.1, 0.5, 0.2, 0, width=0.01, 
         head_width=0.05, color='red', label='Entropie (δS)')
ax.arrow(0.5, 0.1, 0, 0.2, width=0.01,
         head_width=0.05, color='green', label='PID korekce')

# Popisky a titul
plt.title("Kosmická PID regulace\n5D entropická gravitace", fontsize=14, pad=20)
plt.legend(loc='upper right', framealpha=1)
plt.axis('equal')
plt.axis('off')

# Uložení s optimálními parametry
plt.savefig("../cosmic_pid_system.png", 
           dpi=300, 
           bbox_inches='tight',
           format='png',
           transparent=True)

print("Diagram úspěšně vygenerován: visuals/cosmic_pid_system.png")
