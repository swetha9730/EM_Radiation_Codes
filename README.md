# EM_Radiation_Codes

## ⚡ Radiation from an Accelerating Charge

This project simulates and analyzes electromagnetic radiation emitted by a point charge undergoing **oscillatory** and **circular (synchrotron-like)** motion. It uses the **Liénard–Wiechert potentials** to numerically compute:

- Retarded time
- Electric and magnetic fields **E**, **B**
- Poynting vector and energy flux
- Angular power distributions *dP/dΩ*
- Radiation zone residuals and spectral components

---

## 🧪 Features

- Accurate solver for **retarded time** via `scipy.optimize.fsolve`
- Computation of **E** and **B** via:
  - Liénard–Wiechert expressions (exact)
  - Finite differencing (FD)
- Visualizations:
  - Time-evolving field strengths
  - Poynting vector projections
  - Angular power plots and deviations from sin²θ
  - Newtonian vs relativistic emission
  - Synchrotron radiation profiles

---

## 📁 Files

- `EM_radiation_code.py` — core computational functions
- `analysis_script.py` — plots and analysis
- `Radiation_Code_Analysis.pdf` — report summarizing the project

---

## 🌀 Charge Motion Models

- **Linear Oscillation**: along the z-axis  
- **Circular Motion**: in the xy-plane at relativistic speeds (synchrotron)
