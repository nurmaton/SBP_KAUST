import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#############################################
# 1. SBP Matrices using LGL nodes (Reference)
#############################################
class LGL_SBP:
    """
    Computes the Legendre–Gauss–Lobatto nodes, quadrature weights, and the
    differentiation matrix D (and derived matrices P, Q) on the reference element [-1,1].
    """
    def __init__(self, p):
        self.p = p
        self.nodes, self.weights = self._compute_nodes_weights()
        self.D = self._differentiation_matrix()
        self.P = np.diag(self.weights)
        # Compute Q: Q[i,j] = D[i,j] * weights[i]
        self.Q = np.zeros((p+1, p+1))
        for i in range(p+1):
            for j in range(p+1):
                self.Q[i, j] = self.D[i, j] * self.weights[i]

    @staticmethod
    def legendre_poly_coeffs(p):
        poly_dict = {}
        for k in range(p // 2 + 1):
            power = p - 2 * k
            coeff = ((-1)**k * math.comb(p, k) * math.comb(2*p - 2*k, p)) / (2**p)
            poly_dict[power] = coeff
        coeffs = [poly_dict.get(power, 0) for power in range(p, -1, -1)]
        return np.array(coeffs)
    
    @classmethod
    def legendre_poly(cls, p):
        coeffs = cls.legendre_poly_coeffs(p)
        return np.poly1d(coeffs)
    
    def _compute_nodes_weights(self):
        P_poly = self.legendre_poly(self.p)
        dP = P_poly.deriv()
        # The interior nodes are the zeros of P'_p(x)
        interior_nodes = np.sort(dP.r.real)
        # Include endpoints -1 and 1.
        nodes = np.concatenate(([-1.0], interior_nodes, [1.0]))
        # Quadrature weights on [-1,1]:
        weights = 2 / (self.p * (self.p + 1) * (P_poly(nodes)**2))
        return nodes, weights
    
    def _differentiation_matrix(self):
        x = self.nodes
        N = len(x)
        D = np.zeros((N, N))
        b = np.zeros(N)
        # Compute barycentric weights:
        for j in range(N):
            b[j] = 1.0 / np.prod(x[j] - np.delete(x, j))
        # Off-diagonal entries:
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = (b[j] / b[i]) / (x[i] - x[j])
            D[i, i] = -np.sum(D[i, :])
        return D

#############################################
# 2. Grid and SBP assembly in 1D (Cell-Based)
#############################################
class Grid1D:
    """
    Sets up the spatial grid for a 1D problem.
    Divides the domain [a, b] into a given number of cells.
    Each cell is discretized using LGL collocation points (SBP).
    """
    def __init__(self, a, b, num_cells, p, Neqs=1):
        self.a = a
        self.b = b
        self.num_cells = num_cells
        self.p = p           # Polynomial order (each cell has p+1 points)
        self.Neqs = Neqs     # Number of equations (system size per collocation point)
        
        # Compute cell boundaries (uniform grid)
        self.cell_boundaries = np.linspace(a, b, num_cells + 1)
        # Setup SBP matrices on the reference cell [-1,1]
        self.sbp = LGL_SBP(p)
        # Assemble global grid: for each cell, map the reference LGL nodes to the physical cell.
        self.global_nodes = []  # list of arrays (each of shape (p+1,))
        self.cell_sizes = []
        for i in range(num_cells):
            x_left = self.cell_boundaries[i]
            x_right = self.cell_boundaries[i+1]
            self.cell_sizes.append(x_right - x_left)
            # Mapping: y = ((x_right - x_left)/2) * (xi + 1) + x_left
            y = ((x_right - x_left)/2.0) * (self.sbp.nodes + 1) + x_left
            self.global_nodes.append(y)
        # Optionally, assemble a single vector of all nodes (note: cell interfaces appear twice)
        self.all_nodes = np.concatenate(self.global_nodes)
    
    def set_initial_condition(self, init_func):
        """
        Set the initial condition on the grid.
        init_func: callable that accepts an array of coordinates and returns the initial values.
        Returns an array of shape (num_cells, p+1).
        """
        u0 = []
        for cell in self.global_nodes:
            u0.append(init_func(cell))
        u0 = np.array(u0)
        return u0
    
    def apply_periodic_bc(self, u):
        """
        Enforce periodic boundary conditions.
        
        Here, since the grid is assembled cell-by-cell,
        the interfaces between cells appear twice (once as the right endpoint of one cell
        and once as the left endpoint of the next cell). To enforce periodicity,
        we average the duplicate values at each interface and ensure the first and last
        nodes are consistent.
        """
        u_new = u.copy()
        num_cells = self.num_cells
        
        # For internal interfaces: average the right boundary of cell i with the left boundary of cell i+1.
        for i in range(num_cells - 1):
            avg = 0.5 * (u_new[i, -1] + u_new[i+1, 0])
            u_new[i, -1] = avg
            u_new[i+1, 0] = avg
        
        # For the periodic boundary at the domain boundaries:
        avg = 0.5 * (u_new[-1, -1] + u_new[0, 0])
        u_new[-1, -1] = avg
        u_new[0, 0] = avg
        
        return u_new

#############################################
# 3. Compute Spatial Derivative at LGL points
#############################################
def compute_spatial_derivative(u, grid):
    """
    Compute the spatial derivative in each cell.
    u: array of shape (num_cells, p+1) containing solution values at collocation points.
    Returns an array of the same shape containing du/dx.
    """
    num_cells = grid.num_cells
    p = grid.p
    du = np.zeros_like(u)
    for i in range(num_cells):
        # For cell i, map from [-1,1] to physical cell of length dx
        dx = grid.cell_sizes[i]
        # Scale the reference differentiation matrix:
        D_cell = (2 / dx) * grid.sbp.D
        # Compute derivative in cell i
        for j in range(p+1):
            du[i, j] = np.dot(D_cell[j, :], u[i, :])
    return du

#############################################
# 4. ODE Solvers for Time Integration
#############################################
def solve_ode(f, u0, t0, tf, dt, method="RK4", max_iter_BE=50, tol_BE=1e-6):
    """
    Solve du/dt = f(t,u) with initial condition u0 from t0 to tf using step dt.
    Supported methods: "FE" (Forward Euler), "Heun", "RK4", "BE" (Backward Euler).
    Returns time levels and a list of solution arrays (with same shape as u0).
    """
    u_shape = u0.shape
    u = u0.flatten()
    t_values = [t0]
    u_values = [u.copy()]
    t = t0
    while t < tf - 1e-12:
        if t + dt > tf:
            dt = tf - t
        if method == "FE":
            u = u + dt * f(t, u)
        elif method == "Heun":
            f_n = f(t, u)
            u_predict = u + dt * f_n
            f_np1 = f(t+dt, u_predict)
            u = u + dt/2.0 * (f_n + f_np1)
        elif method == "RK4":
            k1 = f(t, u)
            k2 = f(t + dt/2.0, u + dt/2.0 * k1)
            k3 = f(t + dt/2.0, u + dt/2.0 * k2)
            k4 = f(t + dt, u + dt * k3)
            u = u + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        elif method == "BE":
            u_new = u.copy()
            for _ in range(max_iter_BE):
                u_prev = u_new.copy()
                u_new = u + dt * f(t+dt, u_new)
                if np.linalg.norm(u_new - u_prev) < tol_BE:
                    break
            u = u_new
        else:
            raise ValueError("Unknown time integration method.")
        t += dt
        t_values.append(t)
        u_values.append(u.copy())
    u_values = [u_vec.reshape(u_shape) for u_vec in u_values]
    return np.array(t_values), u_values

#############################################
# 5. Main: Assemble, Solve, and Animate a 1D Problem
#############################################
def main():
    # Grid and problem parameters:
    a = 0.0
    b = 1.0
    num_cells = 10       # number of cells
    p = 4                # polynomial order (each cell has 5 points)
    Neqs = 1             # scalar problem
    
    # Create grid:
    grid = Grid1D(a, b, num_cells, p, Neqs)
    
    # Print SBP matrices (from reference element):
    print("Mass matrix P on reference element [-1,1]:")
    print(grid.sbp.P)
    print("\nMatrix Q on reference element [-1,1]:")
    print(grid.sbp.Q)
    
    # Set initial condition (e.g., u(x) = sin(2*pi*x)):
    def init_func(x):
        return np.sin(2*np.pi*x)
    u0 = grid.set_initial_condition(init_func)
    
    # Apply periodic boundary conditions:
    u0 = grid.apply_periodic_bc(u0)
    
    # Define the ODE system (linear advection: u_t = -u_x)
    # (For demonstration, we use a constant speed and simple derivative computation.)
    def ode_system(t, u_vec):
        u_mat = u_vec.reshape(u0.shape)
        du = compute_spatial_derivative(u_mat, grid)
        return (-du).flatten()
    
    # For testing purposes, you might uncomment one of the following:
    # def ode_system(t, u_vec):
    #     # u_t = 0 everywhere (stationary solution)
    #     return np.zeros_like(u_vec)
    # def ode_system(t, u_vec):
    #     # u_t = 1 everywhere (solution should grow linearly in time)
    #     return np.ones_like(u_vec)
    
    # Time integration parameters:
    t0 = 0.0
    tf = 0.5
    dt = 0.001
    # Solve using RK4 (alternatively "FE", "Heun", or "BE")
    t_values, u_values = solve_ode(ode_system, u0, t0, tf, dt, method="BE")
    
    # Animation: Plot each cell's solution as a separate line.
    fig, ax = plt.subplots()
    ax.set_xlim(a, b)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Time-dependent solution (linear advection)")
    
    # Create a list of Line2D objects, one per cell.
    lines = []
    for cell in grid.global_nodes:
        (line,) = ax.plot(cell, np.zeros_like(cell), 'b.-')
        lines.append(line)
    
    # Update function for the animation:
    def update(frame):
        sol = u_values[frame]  # shape: (num_cells, p+1)
        for i, line in enumerate(lines):
            line.set_data(grid.global_nodes[i], sol[i, :])
        ax.set_title(f"Time t = {t_values[frame]:.3f}")
        return lines

    anim = FuncAnimation(fig, update, frames=len(t_values), interval=50, blit=False)
    plt.show()

if __name__ == '__main__':
    main()
