# -*- coding: utf-8 -*-
"""eikonal_1D.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WDCZbs-asoG_exqXYOkAcPqviLGtLjr9
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Parameters
tol         = 1e-6            # --- Convergence tolerance
N           = 101             # --- Number of discretization points
xmin, xmax  = -1, 1           # --- Boundaries of the solution domain

# --- Domain discretization
x, h        = np.linspace(xmin, xmax, N, retstep = True)

# --- Solution initialization
u           = np.zeros(N)
uold        = np.zeros(N)

# --- Iterations
err = np.inf
while err > tol:
    uold[:] = u

    # --- Sweep from left to right
    for n in range(1, N - 1):
        u[n] = min(u[n - 1], u[n + 1]) + h

    # --- Sweep from right to left
    for n in range(N - 2, 0, -1):
        u[n] = min(u[n - 1], u[n + 1]) + h

    err = np.linalg.norm(u - uold)

    plt.figure(1)
    plt.plot(x, np.abs(u))
    plt.draw()
    plt.pause(0.01)

plt.figure(1)
plt.plot(x, np.abs(u))
plt.xlabel('x')
plt.ylabel('u')
plt.show()