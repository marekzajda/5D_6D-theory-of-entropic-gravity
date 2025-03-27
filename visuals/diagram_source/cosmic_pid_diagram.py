import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow

HEAD
fig, ax = plt.subplots(figsize=(10, 6))

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
 77382d7c8565852e174a1ef6ac974f0da6f8688c

# 5D prostor
bulk = Circle((0.5, 0.5), 0.3, fill=False, color='blue', linewidth=3)
ax.add_patch(bulk)

# Holografická plocha
screen = Circle((0.5, 0.5), 0.2, fill=True, color='blue', alpha=0.1)
ax.add_patch(screen)

# Šipky
ax.arrow(0.1, 0.5, 0.2, 0, width=0.01, color='red', label='Entropie (δS)')
ax.arrow(0.5, 0.1, 0, 0.2, width=0.01, color='green', label='PID korekce')

<<<<<<< HEAD
plt.title("Kosmická PID regulace", fontsize=14)
plt.legend()
plt.axis('equal')
plt.axis('off')

plt.savefig("../cosmic_pid_system.png", bbox_inches='tight', dpi=300)
print("Diagram úspěšně vytvořen: ../cosmic_pid_system.png")
=======
plt.title("Kosmická PID regulace", fontsize=14, pad=20)
plt.legend(loc='upper right')
plt.axis('equal')
plt.axis('off')

plt.savefig("../cosmic_pid_system.png", bbox_inches='tight', transparent=True)
print("Diagram vygenerován: ../cosmic_pid_system.png")
>>>>>>> 77382d7c8565852e174a1ef6ac974f0da6f8688c
