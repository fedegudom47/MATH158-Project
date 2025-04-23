import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

os.makedirs("results", exist_ok=True)

# Load green data (in yards, already filtered and converted)
df = pd.read_csv("data/green_under30_yards.csv")
X = df[['holedis']].values
y = df['strokes_remaining'].values  # or 'avg_strokes' if you’ve averaged already

# Define kernel with bounds for smoothness
kernel = RBF(length_scale_bounds=(5, 100)) + WhiteKernel(noise_level=0.05, noise_level_bounds=(0.05, 1.0))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True)
gpr.fit(X, y)

print(f"Optimised kernel for green: {gpr.kernel_}")

# Prediction
X_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_pred, std_pred = gpr.predict(X_grid, return_std=True)

preds = pd.DataFrame({
    'holedis': X_grid.flatten(),
    'pred': y_pred,
    'std': std_pred
})
preds.to_csv("results/gpr_green_preds.csv", index=False)
print("Saved GPR predictions for green.")
