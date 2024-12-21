---
layout: default
title: Solving the 2D Steady-State Heat Equation with CPU and GPU using Numba
---

<script type="text/javascript">
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
  }
};
</script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js">
</script>

In this post, we demonstrate solving the 2D heat equation using the Gauss-Seidel iterative method, implemented both on the CPU and GPU. The GPU implementation leverages the Numba library to accelerate computations.

### The Heat Equation

The 2D steady-state heat equation is given by the partial differential equation:

$$ \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0 $$

where $T(x, y)$ is the temperature at a point $(x, y)$ in the domain.

This equation describes how heat diffuses in a 2D plate under steady-state conditions, assuming no internal heat sources.

In discretized form, using a uniform grid spacing $\Delta x = \Delta y$, the equation becomes:

$$ T_{i,j} = \frac{1}{4} \left( T_{i-1,j} + T_{i+1,j} + T_{i,j-1} + T_{i,j+1} \right) $$

This update formula is used iteratively to compute the temperature distribution over the grid.

---

### Gauss-Seidel Iterations

The Gauss-Seidel method is an iterative solver for systems of linear equations. For the heat equation, the method updates the temperature values grid point by grid point:

1. **Red Points Update**: Update points where $(i + j) \mod 2 = 0$.
2. **Black Points Update**: Update points where $(i + j) \mod 2 = 1$.

This alternating update (red-black ordering) ensures a more efficient convergence compared to updating all points simultaneously thanks to the possibility of parallelizing the two updates, for example, over two different GPUs.

### Code Explanation

The Python implementation of the heat equation solver is split into the following sections:

#### Parameters and Initialization

We define the grid size, maximum iterations, convergence tolerance, and initialize the temperature grid with boundary conditions:

```python
nx, ny = 50, 50  # Grid points
max_iter = 10000  # Max iterations
tolerance = 1e-5  # Convergence tolerance

h_T = np.zeros((ny, nx), dtype=np.float64)  # Temperature grid

# Boundary conditions
h_T[0, :] = h_T[-1, :] = h_T[:, 0] = h_T[:, -1] = 100
```

#### CUDA Kernel for Red-Black Updates

The `red_black_kernel` is a Numba CUDA kernel that updates either red or black points based on the parity of $i + j$:

```python
@cuda.jit
def red_black_kernel(d_T, d_T_old, nx, ny, is_red):
    i, j = cuda.grid(2)
    if 1 <= i < ny - 1 and 1 <= j < nx - 1:
        if (i + j) % 2 == is_red:
            d_T[i, j] = 0.25 * (d_T_old[i-1, j] + d_T_old[i+1, j] + d_T_old[i, j-1] + d_T_old[i, j+1])
```

#### CPU Implementation of Red-Black Updates

The CPU implementation uses nested loops for the red and black updates:

```python
def red_black(h_T, h_T_old, nx, ny, is_red):
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            if (i + j) % 2 == is_red:
                h_T[i, j] = 0.25 * (h_T_old[i-1, j] + h_T_old[i+1, j] + h_T_old[i, j-1] + h_T_old[i, j+1])
```

#### Iterative Solver

The main loop alternates between red and black updates until convergence is achieved:

```python
for iteration in range(max_iter):
    h_T_old = h_T.copy()
    
    # GPU red-black updates
    red_black_kernel[blocks_per_grid, threads_per_block](d_T, d_T_old, nx, ny, 0)
    cuda.synchronize()
    red_black_kernel[blocks_per_grid, threads_per_block](d_T, d_T_old, nx, ny, 1)
    cuda.synchronize()

    # CPU red-black updates
    red_black(h_T, h_T_old, nx, ny, 0)
    red_black(h_T, h_T_old, nx, ny, 1)

    max_diff = np.max(np.abs(h_T - h_T_old))
    if max_diff < tolerance:
        break
```

#### Visualization and Comparison

Finally, the solution is visualized and compared between the CPU and GPU implementations:

```python
h_T_GPU = d_T.copy_to_host()

plt.imshow(h_T, origin='lower', cmap='hot', extent=[0, nx, 0, ny])
plt.title('Temperature Distribution - CPU')
plt.show()

plt.imshow(h_T_GPU, origin='lower', cmap='hot', extent=[0, nx, 0, ny])
plt.title('Temperature Distribution - GPU')
plt.show()

max_diff_CPU_GPU = np.max(np.abs(h_T - h_T_GPU))
rmse = 100. * np.sqrt(np.sum(np.abs(h_T - h_T_GPU)**2) / np.sum(np.abs(h_T)**2))
print(f"Max Difference: {max_diff_CPU_GPU:.2e}")
print(f"RMSE: {rmse:.2e}%")
```
