import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load GPR predictions
df = pd.read_csv("results/gpr_green_preds.csv")
r_vals = df["holedis"].values  # already in yards
z_vals = df["pred"].values

theta = np.linspace(0, 2 * np.pi, 360)
R, T = np.meshgrid(r_vals, theta)
Z = np.tile(z_vals, (len(theta), 1))
X = R * np.cos(T)
Y = R * np.sin(T)

# Plot
plt.figure(figsize=(7, 7))
heat = plt.pcolormesh(X, Y, Z, shading='auto', cmap='inferno_r')

# Concentric rings every 5 yards
for r in range(5, int(r_vals.max()) + 5, 5):
    plt.gca().add_patch(plt.Circle((0, 0), r, color='black', fill=False, linewidth=0.8, linestyle='--'))

plt.text(0, 0, "Pin", ha='center', va='center', fontsize=12, fontweight='bold')
plt.colorbar(heat, label='Predicted strokes to hole out')
plt.title("GPR Heatmap – Green (Putts ≤ 90 feet / 30 yards)")
plt.axis('equal')
plt.xlim(-r_vals.max(), r_vals.max())
plt.ylim(-r_vals.max(), r_vals.max())
plt.tight_layout()
plt.savefig("results/gpr_heatmap_green.png")
plt.show()
