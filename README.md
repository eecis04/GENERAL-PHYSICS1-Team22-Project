# GENERAL-PHYSICS1-Team22-Project

Reproduction of Bradshaw (AJP 2023): projectile motion with quadratic air
resistance, simulated with the 4th-order Runge–Kutta (RK4) method in Python.

## Files
- `experiment1_trajectories.py` — trajectories for launch angles 20°, 40°, 60°, 80° → `Figure_1.png`
- `experiment2_drag_comparison.py` — no-drag vs quadratic-drag comparison + summary table → `Figure_2.png`
- `experiment3_energy.py` — energy verification (K, U, dissipation) → `Figure_3.png`

## How to run
```bash
python experiment1_trajectories.py
python experiment2_drag_comparison.py
python experiment3_energy.py
```
Each script saves its figure as a PNG in the working directory.

## Parameters
m = 0.15 kg, b = 0.0012 kg/m, g = 9.81 m/s², v₀ = 40 m/s

## Requirements
numpy, matplotlib

## Reference
J. L. Bradshaw, "Projectile motion with quadratic drag,"
*American Journal of Physics* **91**, 258–263 (2023).
